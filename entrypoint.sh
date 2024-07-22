#!/bin/bash

echo ${NCCL_PORT_RANGE} > /proc/sys/net/ipv4/ip_local_port_range

torchrun --nproc-per-node=${NPROC_PER_NODE}\
         --nnodes=${NNODES}\
         --rdzv_id=${RDZV_ID}\
         --rdzv-endpoint=${RDZV_ENDPOINT}\
         --node-rank=${NODE_RANK}\
            multinode.py 10 1 config.yaml
