#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import itertools
import logging
from abc import ABC
from typing import Any

import torch
import torch.distributed as dist

import crypten
import crypten.communicator as comm
from crypten.common.rng import generate_kbit_random_tensor, generate_random_ring_element
from crypten.common.util import count_wraps
from crypten.mpc.primitives import ArithmeticSharedTensor, BinarySharedTensor

from .provider import TupleProvider

TTP_FUNCTIONS = ["additive", "square", "binary", "wraps", "B2A"]


class TrustedThirdParty(TupleProvider):
    NAME = "TTP"

    def generate_additive_triple(self, size0, size1, op, device=None, *args, **kwargs):
        """Generate multiplicative triples of given sizes"""
        generator = TTPClient.get().get_generator(device=device)

        a = generate_random_ring_element(size0, generator=generator, device=device)
        b = generate_random_ring_element(size1, generator=generator, device=device)
        if comm.get().get_rank() == 0:
            # Request c from TTP
            c = TTPClient.get().ttp_request("additive", device, size0, size1, op, *args, **kwargs)
        else:
            # TODO: Compute size without executing computation
            c_size = getattr(torch, op)(a, b, *args, **kwargs).size()
            c = generate_random_ring_element(c_size, generator=generator, device=device)

        a = ArithmeticSharedTensor.from_shares(a, precision=0)
        b = ArithmeticSharedTensor.from_shares(b, precision=0)
        c = ArithmeticSharedTensor.from_shares(c, precision=0)

        return a, b, c

    def square(self, size, device=None):
        """Generate square double of given size"""
        generator = TTPClient.get().get_generator(device=device)

        r = generate_random_ring_element(size, generator=generator, device=device)
        if comm.get().get_rank() == 0:
            # Request r2 from TTP
            r2 = TTPClient.get().ttp_request("square", device, size)
        else:
            r2 = generate_random_ring_element(size, generator=generator, device=device)

        r = ArithmeticSharedTensor.from_shares(r, precision=0)
        r2 = ArithmeticSharedTensor.from_shares(r2, precision=0)
        return r, r2

    def generate_binary_triple(self, size0, size1, device=None):
        """Generate binary triples of given size"""
        generator = TTPClient.get().get_generator(device=device)

        a = generate_kbit_random_tensor(size0, generator=generator, device=device)
        b = generate_kbit_random_tensor(size1, generator=generator, device=device)

        if comm.get().get_rank() == 0:
            # Request c from TTP
            c = TTPClient.get().ttp_request("binary", device, size0, size1)
        else:
            size2 = torch.broadcast_tensors(a, b)[0].size()
            c = generate_kbit_random_tensor(size2, generator=generator, device=device)

        # Stack to vectorize scatter function
        a = BinarySharedTensor.from_shares(a)
        b = BinarySharedTensor.from_shares(b)
        c = BinarySharedTensor.from_shares(c)
        return a, b, c

    def wrap_rng(self, size, device=None):
        """Generate random shared tensor of given size and sharing of its wraps"""
        generator = TTPClient.get().get_generator(device=device)

        r = generate_random_ring_element(size, generator=generator, device=device)
        if comm.get().get_rank() == 0:
            # Request theta_r from TTP
            theta_r = TTPClient.get().ttp_request("wraps", device, size)
        else:
            theta_r = generate_random_ring_element(size, generator=generator, device=device)

        r = ArithmeticSharedTensor.from_shares(r, precision=0)
        theta_r = ArithmeticSharedTensor.from_shares(theta_r, precision=0)
        return r, theta_r

    def B2A_rng(self, size, device=None):
        """Generate random bit tensor as arithmetic and binary shared tensors"""
        generator = TTPClient.get().get_generator(device=device)

        # generate random bit
        rB = generate_kbit_random_tensor(size, bitlength=1, generator=generator, device=device)

        if comm.get().get_rank() == 0:
            # Request rA from TTP
            rA = TTPClient.get().ttp_request("B2A", device, size)
        else:
            rA = generate_random_ring_element(size, generator=generator, device=device)

        rA = ArithmeticSharedTensor.from_shares(rA, precision=0)
        rB = BinarySharedTensor.from_shares(rB)
        return rA, rB

    @staticmethod
    def _init():
        TTPClient._init()

    @staticmethod
    def uninit():
        TTPClient.uninit()


class TTPClient:
    __instance = None

    class __TTPClient:
        """Singleton class"""

        def __init__(self):
            # Initialize connection
            self.ttp_group = comm.get().ttp_group
            self.comm_group = comm.get().ttp_comm_group
            self._setup_generators()
            logging.info(f"TTPClient {comm.get().get_rank()} initialized")

        def _setup_generators(self):
            """Setup RNG generator shared between each party (client) and the TTPServer"""
            seed = torch.empty(size=(), dtype=torch.long)
            dist.irecv(tensor=seed, src=comm.get().get_ttp_rank(), group=self.ttp_group).wait()
            dist.barrier(group=self.ttp_group)

            self.generator = torch.Generator(device="cpu")
            self.generator.manual_seed(seed.item())

            if torch.cuda.is_available():
                self.generator_cuda = torch.Generator(device="cuda")
                self.generator_cuda.manual_seed(seed.item())
            else:
                self.generator_cuda = None

        def get_generator(self, device=None):
            if device is None:
                device = "cpu"
            device = torch.device(device)
            if device.type == "cuda":
                return self.generator_cuda
            else:
                return self.generator

        def ttp_request(self, func_name, device, *args, **kwargs) -> torch.LongTensor:
            assert comm.get().get_rank() == 0, "Only party 0 communicates with the TTPServer"
            if device is not None:
                device = str(device)
            message = {
                "function": func_name,
                "device": device,
                "args": args,
                "kwargs": kwargs,
            }
            ttp_rank = comm.get().get_ttp_rank()

            comm.get().send_obj(message, ttp_rank, self.ttp_group)

            size = comm.get().recv_obj(ttp_rank, self.ttp_group)
            result = torch.empty(size, dtype=torch.long, device=device)
            comm.get().broadcast(result, ttp_rank, self.comm_group)

            return result

        def batched_ttp_request(
            self, request_args_list: list[dict[str, Any]]
        ) -> list[torch.LongTensor]:
            assert comm.get().get_rank() == 0, "Only party 0 communicates with the TTPServer"
            message = {
                "function": "batched",
                "device": None,
                "args": request_args_list,
                "kwargs": {},
            }
            ttp_rank = comm.get().get_ttp_rank()

            comm.get().send_obj(message, ttp_rank, self.ttp_group)

            sizes: list[torch.Size] = comm.get().recv_obj(ttp_rank, self.ttp_group)
            assert len(sizes) == len(request_args_list)
            total_sizes: int = sum(size.numel() for size in sizes)
            result_buf = torch.empty(total_sizes, dtype=torch.long, device="cpu")
            comm.get().broadcast(result_buf, ttp_rank, self.comm_group)

            #
            results = []
            for size, request_args in zip(sizes, request_args_list):
                result = result_buf[: size.numel()].view(size)
                result = result.to(device=request_args["device"])
                result_buf = result_buf[size:]
                results.append(result)

            return results

    @staticmethod
    def _init():
        """Initializes a Trusted Third Party client that sends requests"""
        if TTPClient.__instance is None:
            TTPClient.__instance = TTPClient.__TTPClient()

    @staticmethod
    def uninit():
        """Uninitializes a Trusted Third Party client"""
        del TTPClient.__instance
        TTPClient.__instance = None

    @staticmethod
    def get() -> "__TTPClient":
        """Returns the instance of the TTPClient"""
        if TTPClient.__instance is None:
            raise RuntimeError("TTPClient is not initialized")

        return TTPClient.__instance


class TTPServer:
    TERMINATE = -1

    def __init__(self):
        """Initializes a Trusted Third Party server that receives requests"""
        # Initialize connection
        crypten.init()
        self.ttp_group = comm.get().ttp_group
        self.comm_group = comm.get().ttp_comm_group
        self.device = "cpu"
        self._setup_generators()
        ttp_rank = comm.get().get_ttp_rank()

        logging.info("TTPServer Initialized")
        try:
            while True:
                # Wait for next request from client
                message = comm.get().recv_obj(0, self.ttp_group)
                logging.info("Message received: %s" % message)

                if message == "terminate":
                    logging.info("TTPServer shutting down.")
                    return
                elif message["function"] == "batched":
                    # batchify
                    results: list[torch.LongTensor] = []
                    sizes: list[torch.Size] = []
                    request_args_list: list[dict[str, Any]] = message["args"]
                    for request_args in request_args_list:
                        function = request_args["function"]
                        device = request_args["device"]
                        args = request_args["args"]
                        kwargs = request_args["kwargs"]
                        self.device = device

                        result: torch.LongTensor = getattr(self, function)(*args, **kwargs)
                        sizes.append(result.size())
                        results.append(result.to(device="cpu").flatten())
                    comm.get().send_obj(sizes, 0, self.ttp_group)
                    comm.get().broadcast(torch.cat(results, dim=0), ttp_rank, self.comm_group)
                else:
                    # single instruction
                    function = message["function"]
                    device = message["device"]
                    args = message["args"]
                    kwargs = message["kwargs"]

                    self.device = device

                    result = getattr(self, function)(*args, **kwargs)

                    comm.get().send_obj(result.size(), 0, self.ttp_group)
                    comm.get().broadcast(result, ttp_rank, self.comm_group)
        except RuntimeError as err:
            logging.info("Encountered Runtime error. TTPServer shutting down:")
            logging.info(f"{err}")

    def _setup_generators(self):
        """Create random generator to send to a party"""
        ws = comm.get().get_world_size()

        seeds = [torch.randint(-(2**63), 2**63 - 1, size=()) for _ in range(ws)]
        reqs = [dist.isend(tensor=seeds[i], dst=i, group=self.ttp_group) for i in range(ws)]
        self.generators = [torch.Generator(device="cpu") for _ in range(ws)]
        self.generators_cuda = [
            (torch.Generator(device="cuda") if torch.cuda.is_available() else None)
            for _ in range(ws)
        ]

        for i in range(ws):
            self.generators[i].manual_seed(seeds[i].item())
            if torch.cuda.is_available():
                self.generators_cuda[i].manual_seed(seeds[i].item())
            reqs[i].wait()

        dist.barrier(group=self.ttp_group)

    def _get_generators(self, device=None):
        if device is None:
            device = "cpu"
        device = torch.device(device)
        if device.type == "cuda":
            return self.generators_cuda
        else:
            return self.generators

    def _get_additive_PRSS(self, size, remove_rank=False):
        """
        Generates a plaintext value from a set of random additive secret shares
        generated by each party
        """
        gens = self._get_generators(device=self.device)
        if remove_rank:
            gens = gens[1:]
        result = None
        for idx, g in enumerate(gens):
            elem = generate_random_ring_element(size, generator=g, device=g.device)
            result = elem if idx == 0 else result + elem
        return result

    def _get_binary_PRSS(self, size, bitlength=None, remove_rank=None):
        """
        Generates a plaintext value from a set of random binary secret shares
        generated by each party
        """
        gens = self._get_generators(device=self.device)
        if remove_rank:
            gens = gens[1:]
        result = None
        for idx, g in enumerate(gens):
            elem = generate_kbit_random_tensor(
                size, bitlength=bitlength, generator=g, device=g.device
            )
            result = elem if idx == 0 else result ^ elem
        return result

    def additive(self, size0, size1, op, *args, **kwargs):
        # Add all shares of `a` and `b` to get plaintext `a` and `b`
        a = self._get_additive_PRSS(size0)
        b = self._get_additive_PRSS(size1)

        c = getattr(torch, op)(a, b, *args, **kwargs)

        # Subtract all other shares of `c` from plaintext value of `c` to get `c0`
        c0 = c - self._get_additive_PRSS(c.size(), remove_rank=True)
        return c0

    def square(self, size):
        # Add all shares of `r` to get plaintext `r`
        r = self._get_additive_PRSS(size)
        r2 = r.mul(r)
        return r2 - self._get_additive_PRSS(size, remove_rank=True)

    def multi_mul(self, n: int, size: torch.Size, *args, **kwargs):
        """Generate shares for rank 0 of the high order terms."""
        # TODO
        num_terms = 2**n - 1
        order_one_terms = [self._get_additive_PRSS(size) for _ in range(n)]
        high_order_terms = []
        for order in range(2, n + 1):
            order_indices = list(itertools.combinations(range(n), order))
            for indices in order_indices:
                plain: torch.Tensor = functools.reduce(
                    lambda x, y: torch.mul(x, y), [order_one_terms[i] for i in indices]
                )
                tmp = plain - self._get_additive_PRSS(size, remove_rank=True)
                high_order_terms.append(tmp)
        return torch.stack(high_order_terms, dim=0)

    def binary(self, size0, size1):
        # xor all shares of `a` and `b` to get plaintext `a` and `b`
        a = self._get_binary_PRSS(size0)
        b = self._get_binary_PRSS(size1)

        c = a & b

        # xor all other shares of `c` from plaintext value of `c` to get `c0`
        c0 = c ^ self._get_binary_PRSS(c.size(), remove_rank=True)
        return c0

    def wraps(self, size):
        r = [generate_random_ring_element(size, generator=g) for g in self.generators]
        theta_r = count_wraps(r)

        return theta_r - self._get_additive_PRSS(size, remove_rank=True)

    def B2A(self, size):
        rB = self._get_binary_PRSS(size, bitlength=1)

        # Subtract all other shares of `rA` from plaintext value of `rA`
        rA = rB - self._get_additive_PRSS(size, remove_rank=True)

        return rA


#################
#               #
#    Actions    #
#               #
#################
class TTPAction(ABC):
    def pre_request_stage(self) -> Any:
        """Prepares for the request"""
        ...

    def ttp_request_args(self) -> dict[str, Any] | None:
        """Args of TTP requrests. If None, no communication is needed."""
        ...

    def post_request_stage(self, pre_stage_output: Any, ttp_request_result: torch.LongTensor):
        """Post processing of the request"""
        ...

    def wait(self) -> None:
        """Waits for the single request to complete"""
        if self.is_completed():
            return
        # pre stage
        pre_stage_output = self.pre_request_stage()
        # merge ttp request
        request_args = self.ttp_request_args()
        if request_args is not None:
            ttp_result = TTPClient.get().ttp_request(**request_args)
        else:
            ttp_result = None
        # post stage
        self.post_request_stage(pre_stage_output, ttp_result)

    def is_completed(self) -> bool:
        """Checks if the request is completed"""
        ...

    def get_result(self) -> Any:
        """Returns the result of the request"""
        ...

    def group(self, *actions: "TTPAction") -> "TTPActionGroup":
        """Groups this action with other actions"""
        actions = (self, *actions)
        return TTPActionGroup(self, *actions)


class DummyTTPAction(TTPAction):
    def __init__(self, result=None):
        self.result = result

    def pre_request_stage(self) -> Any:
        """Prepares for the request"""
        return None

    def ttp_request_args(self) -> dict[str, Any] | None:
        """Args of TTP requrests"""
        return None

    def post_request_stage(self, pre_stage_output: Any, ttp_request_result: torch.LongTensor):
        """Post processing of the request"""
        pass

    def is_completed(self) -> bool:
        """Checks if the request is completed"""
        return True

    def get_result(self):
        """Returns the result of the request"""
        return self.result


class GenAddTripleTTPAction(TTPAction):
    def __init__(
        self,
        size0: torch.Size,
        size1: torch.Size,
        op: str,
        device: torch.device | str | None = None,
        *args,
        **kwargs,
    ):
        """Action for generating an additive triple.

        Args:
            size0 (int): size of the first operand
            size1 (int): size of the second operand
            op (str): operation to perform on the operands
            device (torch.device | str | None): device to perform the operation on
            *args: additional arguments
            **kwargs: additional keyword arguments
        """
        self.size0 = size0
        self.size1 = size1
        self.op = op
        self.device = device
        self.args = args
        self.kwargs = kwargs
        self._result = None

    def pre_request_stage(self) -> Any:
        """Prepares for the request"""
        generator = TTPClient.get().get_generator(device=self.device)

        a = generate_random_ring_element(self.size0, generator=generator, device=self.device)
        b = generate_random_ring_element(self.size1, generator=generator, device=self.device)

        return a, b

    def ttp_request_args(self) -> dict[str, Any] | None:
        """Args of TTP requrests"""
        if comm.get().get_rank() == 0:
            return {
                "func_name": "additive",
                "device": str(self.device) if self.device is not None else None,
                "args": (self.size0, self.size1, self.op, *self.args),
                "kwargs": self.kwargs,
            }
        else:
            return None

    def post_request_stage(self, pre_stage_output: Any, ttp_request_result: torch.LongTensor):
        """Post processing of the request"""
        a, b = pre_stage_output
        if comm.get().get_rank() == 0:
            # Request c from TTP
            c = ttp_request_result
        else:
            # TODO: Compute size without executing computation
            generator = TTPClient.get().get_generator(device=self.device)
            c_size = getattr(torch, self.op)(a, b, *self.args, **self.kwargs).size()
            c = generate_random_ring_element(c_size, generator=generator, device=self.device)

        a = ArithmeticSharedTensor.from_shares(a, precision=0)
        b = ArithmeticSharedTensor.from_shares(b, precision=0)
        c = ArithmeticSharedTensor.from_shares(c, precision=0)

        self._result = (a, b, c)

    def is_completed(self) -> bool:
        """Checks if the request is completed"""
        return self._result is not None

    def get_result(
        self,
    ) -> tuple[ArithmeticSharedTensor, ArithmeticSharedTensor, ArithmeticSharedTensor]:
        """Returns the result of the request"""
        return self._result


class SquareTTPAction(TTPAction):
    def __init__(self, size: torch.Size, device: torch.device | str | None = None):
        """Action for generating a square double.

        Args:
            size (int): size of the operand
            device (torch.device | str | None): device to perform the operation on
        """
        self.size = size
        self.device = device
        self._result = None

    def pre_request_stage(self) -> Any:
        """Prepares for the request"""
        generator = TTPClient.get().get_generator(device=self.device)

        r = generate_random_ring_element(self.size, generator=generator, device=self.device)
        return r

    def ttp_request_args(self) -> dict[str, Any] | None:
        """Args of TTP requrests"""
        if comm.get().get_rank() == 0:
            return {
                "func_name": "square",
                "device": str(self.device) if self.device is not None else None,
                "args": (self.size,),
                "kwargs": {},
            }
        else:
            return None

    def post_request_stage(self, pre_stage_output: Any, ttp_request_result: torch.LongTensor):
        """Post processing of the request"""
        r = pre_stage_output
        if comm.get().get_rank() == 0:
            # Request r2 from TTP
            r2 = ttp_request_result
        else:
            generator = TTPClient.get().get_generator(device=self.device)
            r2 = generate_random_ring_element(self.size, generator=generator, device=self.device)

        r = ArithmeticSharedTensor.from_shares(r, precision=0)
        r2 = ArithmeticSharedTensor.from_shares(r2, precision=0)
        self._result = (r, r2)

    def is_completed(self) -> bool:
        """Checks if the request is completed"""
        return self._result is not None

    def get_result(self) -> tuple[ArithmeticSharedTensor, ArithmeticSharedTensor]:
        """Returns the result of the request"""
        return self._result


class MultiMulTTPAction(TTPAction):
    def __init__(
        self,
        n: int,
        size: torch.Size,
        device: torch.device | str | None = None,
        *args,
        **kwargs,
    ):
        """Action for generating an additive triple.

        Args:
            sizes (int): size of the first operand
            device (torch.device | str | None): device to perform the operation on
            *args: additional arguments
            **kwargs: additional keyword arguments
        """
        self.n = n
        self.num_terms = 2**n - 1
        self.size = size
        self.device = device
        self.args = args
        self.kwargs = kwargs
        self._result = None

    def pre_request_stage(self) -> Any:
        """Prepares for the request"""
        generator = TTPClient.get().get_generator(device=self.device)

        order_one_results: list[torch.Tensor] = []
        high_order_results: list[torch.Tensor] = []

        # first n terms are sampled locally
        for _ in range(self.n):
            order_one_results.append(
                generate_random_ring_element(self.size, generator=generator, device=self.device)
            )

        # rest of the terms are sampled locally, excluding party 0
        if comm.get().get_rank() != 0:
            for _ in range(self.num_terms - self.n):
                high_order_results.append(
                    generate_random_ring_element(self.size, generator=generator, device=self.device)
                )

        return torch.stack(order_one_results, dim=0), torch.stack(high_order_results, dim=0)

    def ttp_request_args(self) -> dict[str, Any] | None:
        """Args of TTP requrests"""
        if comm.get().get_rank() == 0:
            return {
                "func_name": "multi_mul",
                "device": str(self.device) if self.device is not None else None,
                "args": (self.n, self.size, *self.args),
                "kwargs": self.kwargs,
            }
        else:
            return None

    def post_request_stage(self, pre_stage_output: Any, ttp_request_result: torch.LongTensor):
        """Post processing of the request"""
        order_one_results: torch.Tensor
        high_order_results: torch.Tensor
        order_one_results, high_order_results = pre_stage_output
        if comm.get().get_rank() == 0:
            # Request c from TTP
            high_order_results = ttp_request_result

        assert order_one_results.size(0) == self.n
        assert high_order_results.size(0) == self.num_terms - self.n
        self._result = []
        for t in itertools.chain(order_one_results, high_order_results):
            self._result.append(ArithmeticSharedTensor.from_shares(t, precision=0))

    def is_completed(self) -> bool:
        """Checks if the request is completed"""
        return self._result is not None

    def get_result(
        self,
    ) -> list[ArithmeticSharedTensor]:
        """Returns the result of the request"""
        return self._result


class TTPActionGroup:
    def __init__(self, *actions: TTPAction) -> None:
        assert all(isinstance(action, TTPAction) for action in actions)
        self.actions = list(actions)

    def wait(self) -> None:
        """Waits for all actions to complete"""
        if all(action.is_completed() for action in self.actions):
            return
        # pre stage
        pre_stage_outputs = [action.pre_request_stage() for action in self.actions]
        # merge ttp request
        request_args_list = [action.ttp_request_args() for action in self.actions]
        if all(request_args is not None for request_args in request_args_list):
            ttp_results = TTPClient.get().batched_ttp_request(request_args_list)
        else:
            ttp_results = [None] * len(self.actions)
        # post stage
        for action, pre_stage_output, ttp_request_result in zip(
            self.actions, pre_stage_outputs, ttp_results
        ):
            action.post_request_stage(pre_stage_output, ttp_request_result)

    def is_completed(self) -> bool:
        return all(action.is_completed() for action in self.actions)

    def get_result(self) -> tuple[Any]:
        return tuple(action.get_result() for action in self.actions)

    def add_(self, *actions: TTPAction) -> "TTPActionGroup":
        assert all(isinstance(action, TTPAction) for action in actions)
        self.actions.extend(actions)
        return self

    def add_group_(self, group: "TTPActionGroup") -> "TTPActionGroup":
        self.actions.extend(group.actions)
        return self
