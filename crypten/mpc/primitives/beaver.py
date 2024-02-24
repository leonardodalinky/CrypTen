#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import itertools
import math
from typing import TYPE_CHECKING

import torch

import crypten
import crypten.communicator as comm
from crypten.common.util import count_wraps
from crypten.config import cfg

if TYPE_CHECKING:
    from crypten.mpc.provider.ttp_provider import (
        GenAddTripleTTPAction,
        MultiMulTTPAction,
        SquareTTPAction,
        WrapsTTPAction,
    )

    from .arithmetic import ArithmeticSharedTensor


class IgnoreEncodings:
    """Context Manager to ignore tensor encodings"""

    def __init__(self, list_of_tensors):
        self.list_of_tensors = list_of_tensors
        self.encodings_cache = [tensor.encoder.scale for tensor in list_of_tensors]

    def __enter__(self):
        for tensor in self.list_of_tensors:
            tensor.encoder._scale = 1

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for i, tensor in enumerate(self.list_of_tensors):
            tensor.encoder._scale = self.encodings_cache[i]


def __beaver_protocol(
    op, x, y, ttp_action: "(GenAddTripleTTPAction | None)" = None, *args, **kwargs
):
    """Performs Beaver protocol for additively secret-shared tensors x and y

    1. Obtain uniformly random sharings [a],[b] and [c] = [a * b]
    2. Additively hide [x] and [y] with appropriately sized [a] and [b]
    3. Open ([epsilon] = [x] - [a]) and ([delta] = [y] - [b])
    4. Return [z] = [c] + (epsilon * [b]) + ([a] * delta) + (epsilon * delta)
    """
    assert op in {
        "mul",
        "matmul",
        "conv1d",
        "conv2d",
        "conv_transpose1d",
        "conv_transpose2d",
    }
    if x.device != y.device:
        raise ValueError(f"x lives on device {x.device} but y on device {y.device}")

    provider = crypten.mpc.get_default_provider()
    if ttp_action is not None and crypten.mpc.low_latency_enabled():
        ttp_action.wait()
        a, b, c = ttp_action.get_result()
    else:
        a, b, c = provider.generate_additive_triple(
            x.size(), y.size(), op, device=x.device, *args, **kwargs
        )

    from .arithmetic import ArithmeticSharedTensor

    if cfg.mpc.active_security:
        """
        Reference: "Multiparty Computation from Somewhat Homomorphic Encryption"
        Link: https://eprint.iacr.org/2011/535.pdf
        """
        f, g, h = provider.generate_additive_triple(
            x.size(), y.size(), op, device=x.device, *args, **kwargs
        )

        t = ArithmeticSharedTensor.PRSS(a.size(), device=x.device)
        t_plain_text = t.get_plain_text()

        rho = (t_plain_text * a - f).get_plain_text()
        sigma = (b - g).get_plain_text()
        triples_check = t_plain_text * c - h - sigma * f - rho * g - rho * sigma
        triples_check = triples_check.get_plain_text()

        if torch.any(triples_check != 0):
            raise ValueError("Beaver Triples verification failed!")

    # Vectorized reveal to reduce rounds of communication
    with IgnoreEncodings([a, b, x, y]):
        epsilon, delta = ArithmeticSharedTensor.reveal_batch([x - a, y - b])

    # z = c + (a * delta) + (epsilon * b) + epsilon * delta
    c._tensor += getattr(torch, op)(epsilon, b._tensor, *args, **kwargs)
    c._tensor += getattr(torch, op)(a._tensor, delta, *args, **kwargs)
    c += getattr(torch, op)(epsilon, delta, *args, **kwargs)

    return c


def mul(x, y, ttp_action: "(GenAddTripleTTPAction | None)" = None):
    return __beaver_protocol("mul", x, y, ttp_action=ttp_action)


def multi_mul(*tensors: "ArithmeticSharedTensor", ttp_action: "(MultiMulTTPAction | None)" = None):
    n = len(tensors)
    assert n >= 2, "multi_mul requires at least 2 tensors"

    if ttp_action is None:
        from crypten.mpc.provider.ttp_provider import MultiMulTTPAction

        ttp_action = MultiMulTTPAction(n, tensors[0]._tensor.size(), tensors[0].device)
    ttp_action.wait()
    terms = ttp_action.get_result()  # length of `2**n - 1`
    assert len(terms) == 2**n - 1, "multi_mul requires 2^n - 1 terms"

    # reveal all delta
    with IgnoreEncodings([*tensors, *terms]):
        from .arithmetic import ArithmeticSharedTensor

        deltas = ArithmeticSharedTensor.reveal_batch(
            [tensor - term for tensor, term in zip(tensors, terms[:n])]
        )  # length of `n`

    # `term_order = n`'s term
    ret: ArithmeticSharedTensor = terms[-1].clone()
    # `term_order = 0`'s term
    # NOTE: SHOULD not use `ret._tensor`, since here we only add a constant to this tensor,
    #       and only rank-0 node should do addition
    ret += math.prod(deltas)
    for term_order in range(1, n):
        term_order_bidx = sum(math.comb(n, i) for i in range(1, term_order))
        term_order_eidx = term_order_bidx + math.comb(n, term_order)
        cur_terms = terms[term_order_bidx:term_order_eidx]
        assert len(cur_terms) == math.comb(n, term_order)
        for term_indices, cur_term in zip(itertools.combinations(range(n), term_order), cur_terms):
            term_indices: list[int]
            cur_term: ArithmeticSharedTensor
            # compute current term to be added to result
            tmp = cur_term._tensor.clone()
            delta_indices = [i for i in range(n) if i not in term_indices]
            assert len(delta_indices) + term_order == n
            for delta_idx in delta_indices:
                tmp = tmp * deltas[delta_idx]
            ret._tensor += tmp

    return ret


def matmul(x, y, ttp_action: "(GenAddTripleTTPAction | None)" = None):
    return __beaver_protocol("matmul", x, y, ttp_action=ttp_action)


def conv1d(x, y, ttp_action: "(GenAddTripleTTPAction | None)" = None, **kwargs):
    return __beaver_protocol("conv1d", x, y, ttp_action=ttp_action, **kwargs)


def conv2d(x, y, ttp_action: "(GenAddTripleTTPAction | None)" = None, **kwargs):
    return __beaver_protocol("conv2d", x, y, ttp_action=ttp_action, **kwargs)


def conv_transpose1d(x, y, ttp_action: "(GenAddTripleTTPAction | None)" = None, **kwargs):
    return __beaver_protocol("conv_transpose1d", x, y, ttp_action=ttp_action, **kwargs)


def conv_transpose2d(x, y, ttp_action: "(GenAddTripleTTPAction | None)" = None, **kwargs):
    return __beaver_protocol("conv_transpose2d", x, y, ttp_action=ttp_action, **kwargs)


def square(x, ttp_action: "(SquareTTPAction | None)" = None):
    """Computes the square of `x` for additively secret-shared tensor `x`

    1. Obtain uniformly random sharings [r] and [r2] = [r * r]
    2. Additively hide [x] with appropriately sized [r]
    3. Open ([epsilon] = [x] - [r])
    4. Return z = [r2] + 2 * epsilon * [r] + epsilon ** 2
    """
    provider = crypten.mpc.get_default_provider()
    if ttp_action is not None and crypten.mpc.low_latency_enabled():
        ttp_action.wait()
        r, r2 = ttp_action.get_result()
    else:
        r, r2 = provider.square(x.size(), device=x.device)

    with IgnoreEncodings([x, r]):
        epsilon = (x - r).reveal()
    return r2 + 2 * r * epsilon + epsilon * epsilon


def wraps(x, ttp_action: "(WrapsTTPAction | None)" = None):
    """Privately computes the number of wraparounds for a set a shares

    To do so, we note that:
        [theta_x] = theta_z + [beta_xr] - [theta_r] - [eta_xr]

    Where [theta_i] is the wraps for a variable i
          [beta_ij] is the differential wraps for variables i and j
          [eta_ij]  is the plaintext wraps for variables i and j

    Note: Since [eta_xr] = 0 with probability 1 - |x| / Q for modulus Q, we
    can make the assumption that [eta_xr] = 0 with high probability.
    """
    if ttp_action is not None and crypten.mpc.low_latency_enabled():
        ttp_action.wait()
        r, theta_r = ttp_action.get_result()
    else:
        provider = crypten.mpc.get_default_provider()
        r, theta_r = provider.wrap_rng(x.size(), device=x.device)
    beta_xr = theta_r.clone()
    beta_xr._tensor = count_wraps([x._tensor, r._tensor])

    with IgnoreEncodings([x, r]):
        z = x + r
    theta_z = comm.get().gather(z._tensor, 0)
    theta_x = beta_xr - theta_r

    # TODO: Incorporate eta_xr
    if x.rank == 0:
        theta_z = count_wraps(theta_z)
        theta_x._tensor += theta_z
    return theta_x


def truncate(x, y, wraps_ttp_action: "(WrapsTTPAction | None)" = None):
    """Protocol to divide an ArithmeticSharedTensor `x` by a constant integer `y`"""
    wrap_count = wraps(x, ttp_action=wraps_ttp_action)
    x.share = x.share.div_(y, rounding_mode="trunc")
    # NOTE: The multiplication here must be split into two parts
    # to avoid long out-of-bounds when y <= 2 since (2 ** 63) is
    # larger than the largest long integer.
    correction = wrap_count * 4 * (int(2**62) // y)
    x.share -= correction.share
    return x


def AND(x, y):
    """
    Performs Beaver protocol for binary secret-shared tensors x and y

    1. Obtain uniformly random sharings [a],[b] and [c] = [a & b]
    2. XOR hide [x] and [y] with appropriately sized [a] and [b]
    3. Open ([epsilon] = [x] ^ [a]) and ([delta] = [y] ^ [b])
    4. Return [c] ^ (epsilon & [b]) ^ ([a] & delta) ^ (epsilon & delta)
    """
    from .binary import BinarySharedTensor

    provider = crypten.mpc.get_default_provider()
    a, b, c = provider.generate_binary_triple(x.size(), y.size(), device=x.device)

    # Stack to vectorize reveal
    eps_del = BinarySharedTensor.reveal_batch([x ^ a, y ^ b])
    epsilon = eps_del[0]
    delta = eps_del[1]

    return (b & epsilon) ^ (a & delta) ^ (epsilon & delta) ^ c


def B2A_single_bit(xB):
    """Converts a single-bit BinarySharedTensor xB into an
        ArithmeticSharedTensor. This is done by:

    1. Generate ArithmeticSharedTensor [rA] and BinarySharedTensor =rB= with
        a common 1-bit value r.
    2. Hide xB with rB and open xB ^ rB
    3. If xB ^ rB = 0, then return [rA], otherwise return 1 - [rA]
        Note: This is an arithmetic xor of a single bit.
    """
    if comm.get().get_world_size() < 2:
        from .arithmetic import ArithmeticSharedTensor

        return ArithmeticSharedTensor(xB._tensor, precision=0, src=0)

    provider = crypten.mpc.get_default_provider()
    rA, rB = provider.B2A_rng(xB.size(), device=xB.device)

    z = (xB ^ rB).reveal()
    rA = rA * (1 - 2 * z) + z
    return rA
