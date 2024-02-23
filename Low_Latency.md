# Low-latency Crypten

We are working on improving the latency of CrypTen. We are currently working on the following:
* Batchify requests to TTP
* Parallelizing complex operations
* Optimizing multi-operand multiplications

## Setup

Enable the low-latency operations in `configs/default.yaml`.
```yaml
mpc:
  low_latency: True
```


## docker-tc

To simulate a high-latency network, we use [docker-tc](https://github.com/lukaszlach/docker-tc).

First, start the `docker-tc` daemon:
```bash
docker run -d \
    --name docker-tc \
    --network host \
    --cap-add NET_ADMIN \
    --restart always \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /var/docker-tc:/var/docker-tc \
    lukaszlach/docker-tc
```

Then check the `--latency` and `bandwidth` options in `docker_launcher.py`.
