import argparse
import logging
import os
import os.path as osp
import signal
import sys
import uuid

import docker
import docker.errors
import docker.types
import torch
from docker.models.containers import Container
from docker.models.images import Image
from docker.models.networks import Network
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s: %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Helper utility that runs multinode scripts on docker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tag-prefix",
        type=str,
        help="The prefix for the docker image tag",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
        help="Number of parties to launch. Each party acts as its own process. "
        "The TTP party is not included in this number.",
    )
    parser.add_argument(
        "--disable-ttp",
        action="store_true",
        help="Disable TTP party",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=29500,
        help="The port used by master instance for distributed training",
    )
    # docker-related
    parser.add_argument(
        "--dockerfile",
        type=str,
        default=osp.join(osp.dirname(__file__), "..", "Dockerfile"),
        help="The directory for dockerfile to build the docker image from",
    )
    parser.add_argument(
        "--ttp-dockerfile",
        type=str,
        default=osp.join(osp.dirname(__file__), "..", "TTP.Dockerfile"),
        help="The directory for dockerfile to build the TTP docker image from",
    )
    parser.add_argument(
        "--script-entrypoint",
        type=str,
        default="launcher.py",
        help="Entrypoint of the script in `script-dir` to be launched",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force to rebuild the docker images of normal and TTP parties",
    )
    # positional
    parser.add_argument(
        "script_dir",
        type=str,
        help="The dir to the program/script to be launched in parallel, "
        "followed by all the arguments for the script",
    )
    # rest from the training program
    parser.add_argument("script_args", nargs=argparse.REMAINDER)

    return parser.parse_args()


def has_image(client: docker.DockerClient, image_name: str) -> bool:
    try:
        client.images.get(image_name)
        return True
    except docker.errors.ImageNotFound:
        return False


def delete_image_if_exists(client: docker.DockerClient, image_name: str) -> None:
    try:
        image: Image = client.images.get(image_name)
        logging.info(f"Removing existing image: {image_name}")
        image.remove(force=True)
    except docker.errors.ImageNotFound:
        pass


class DockerCtx:
    def __init__(
        self,
        client: docker.DockerClient,
        tag_prefix: str,
        world_size: int,
        enable_ttp: bool,
        master_port: int,
        script_dir: str,
        script_entrypoint: str = "launcher.py",
        script_args: list[str] = [],
    ) -> None:
        self.__is_setup__ = False
        self.client = client
        self.tag_prefix = tag_prefix
        self.world_size = world_size
        self.enable_ttp = enable_ttp
        self.master_port = master_port
        self.script_dir = script_dir
        assert osp.isdir(self.script_dir), "Script dir does not exist"
        self.script_entrypoint = script_entrypoint
        self.script_args = script_args
        assert osp.exists(
            osp.join(self.script_dir, self.script_entrypoint)
        ), "Entrypoint file does not exist"
        # docker objs
        self.network: Network | None = None
        self.containers: list[Container] = []
        self.ttp_container: Container | None = None
        # signals
        logging.debug("Hooking SIGINT signal")
        signal.signal(signal.SIGINT, self._sigint_handler)

    def __enter__(self):
        self.setup()

    def __exit__(self, type, value, traceback):
        self.cleanup()

    def setup(self):
        self.__is_setup__ = True
        gpu_count = torch.cuda.device_count()
        logging.info(f"Detect {gpu_count} GPUs.")
        assert gpu_count >= 1, "No GPU detected"
        ########################
        #                      #
        #    create network    #
        #                      #
        ########################
        uid = str(uuid.uuid4()).replace("-", "")
        uid = uid[:8]
        logging.debug(f"Creating network: crypten_{uid}")
        self.network = self.client.networks.create(
            f"{self.tag_prefix}-crypten-{uid}", driver="bridge"
        )
        #####################
        #                   #
        #    run dockers    #
        #                   #
        #####################
        # normal parties
        logging.info("Creating and running docker containers of normal parties...")
        for i in range(self.world_size):
            container_name = f"{self.tag_prefix}-crypten-{uid}-{i}"
            logging.debug(f"Creating and running container: {container_name}")
            container = self.client.containers.run(
                f"{self.tag_prefix}/crypten:latest",
                name=container_name,
                environment={
                    "WORLD_SIZE": str(self.world_size),
                    "RANK": str(i),
                    "RENDEZVOUS": "env://",
                    "BACKEND": "gloo",
                    "MASTER_ADDR": f"{self.tag_prefix}-crypten-{uid}-0",
                    "MASTER_PORT": str(self.master_port),
                },
                entrypoint=["python", self.script_entrypoint],
                command=self.script_args,
                detach=True,
                network=self.network.name,
                user=os.getuid(),
                volumes={
                    osp.abspath(self.script_dir): {
                        "bind": "/app",
                        "mode": "rw",
                    },
                    osp.abspath(osp.join(osp.dirname(__file__), "..", "configs")): {
                        "bind": "/framework/configs",
                        "mode": "ro",
                    },
                },
                device_requests=[
                    docker.types.DeviceRequest(
                        device_ids=[str(i % gpu_count)], capabilities=[["gpu"]]
                    )
                ],
            )
            self.containers.append(container)
        # TTP party
        if self.enable_ttp:
            ttp_container_name = f"{self.tag_prefix}-crypten-ttp-{uid}"
            logging.info("Creating and running docker container of TTP party...")
            logging.debug(f"Creating and running container: {container_name}")
            # TTP image
            self.ttp_container = self.client.containers.run(
                f"{self.tag_prefix}/crypten-ttp:latest",
                name=ttp_container_name,
                environment={
                    "WORLD_SIZE": str(self.world_size),
                    "RANK": str(self.world_size),
                    "RENDEZVOUS": "env://",
                    "BACKEND": "gloo",
                    "MASTER_ADDR": f"{self.tag_prefix}-crypten-{uid}-0",
                    "MASTER_PORT": str(self.master_port),
                },
                detach=True,
                network=self.network.name,
                user=os.getuid(),
                volumes={
                    osp.abspath(osp.join(osp.dirname(__file__), "..", "configs")): {
                        "bind": "/framework/configs",
                        "mode": "ro",
                    },
                },
                device_requests=[
                    docker.types.DeviceRequest(
                        device_ids=[str(gpu_count - 1)], capabilities=[["gpu"]]
                    )
                ],
            )

    def wait(self) -> None:
        """Wait for the first containers to finish and print out logs."""
        container = self.containers[0]
        # refresh status
        for log in container.logs(stream=True, follow=True):
            print(log.decode("utf-8"))

    def get_logs(self, idx: int) -> str:
        """Get logs of the idx-th container."""
        return self.containers[idx].logs().decode("utf-8")

    def cleanup(self):
        if not self.__is_setup__:
            return

        self.__is_setup__ = False
        logging.info("Cleaning up...")
        logging.info("Stopping docker containers of normal parties...")
        for container in self.containers:
            logging.debug(f"Removing container: {container.name}")
            container.remove(link=False, force=True)
        if self.ttp_container:
            logging.info("Stopping docker container of TTP party...")
            logging.debug(f"Removing container: {self.ttp_container.name}")
            self.ttp_container.remove(link=False, force=True)
        if self.network:
            logging.info("Removing docker network...")
            logging.debug(f"Removing network: {self.network.name}")
            self.network.remove()
        else:
            logging.warning("No network to remove")

    def _sigint_handler(self, signal_received, frame):
        logging.info("Gracefully shutdown docker containers...")

        self.__exit__(None, None, None)
        sys.exit(0)


def main():
    assert os.getenv("CUDA_VISIBLE_DEVICES") is None, "CUDA_VISIBLE_DEVICES should not be set"
    args = parse_args()
    client = docker.from_env()
    ######################
    #                    #
    #    build images    #
    #                    #
    ######################
    # normal party
    normal_party_iname = f"{args.tag_prefix}/crypten:latest"
    if not has_image(client, normal_party_iname) or args.force_rebuild:
        logging.info(f"Building docker images of normal parties from: {args.dockerfile}")
        delete_image_if_exists(client, normal_party_iname)
        client.images.build(
            path=osp.dirname(args.dockerfile),
            dockerfile=args.dockerfile,
            tag=normal_party_iname,
        )
    else:
        logging.info(f"Find existing docker image of normal party: {normal_party_iname}")
    # TTP party
    ttp_party_iname = f"{args.tag_prefix}/crypten-ttp:latest"
    if not has_image(client, ttp_party_iname) or args.force_rebuild:
        logging.info("Building docker image of TTP party...")
        delete_image_if_exists(client, ttp_party_iname)
        client.images.build(
            path=osp.dirname(args.ttp_dockerfile),
            dockerfile=args.ttp_dockerfile,
            tag=ttp_party_iname,
        )
    else:
        logging.info(f"Find existing docker image of TTP party: {ttp_party_iname}")

    docker_ctx = DockerCtx(
        client=client,
        tag_prefix=args.tag_prefix,
        world_size=args.world_size,
        enable_ttp=not args.disable_ttp,
        master_port=args.master_port,
        script_dir=args.script_dir,
        script_entrypoint=args.script_entrypoint,
        script_args=args.script_args,
    )
    with docker_ctx:
        logging.info("Running docker containers...")
        logging.info("Press Ctrl+C to stop the containers")
        logging.info("The logs of the first container:")
        docker_ctx.wait()
        input("Press any key to continue...")


if __name__ == "__main__":
    main()
