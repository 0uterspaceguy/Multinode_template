# echo 50000 51000 > /proc/sys/net/ipv4/ip_local_port_range
# ufw allow 50000:51000/tcp

# torchrun --nproc-per-node=1 --nnodes=1 --rdzv_id=456 --rdzv-endpoint=192.168.0.35:12345 --rdzv-backend=c10d --node-rank=1 multinode.py 10 1 config.yaml
torchrun --nproc-per-node=1 --nnodes=2 --rdzv_id=456 --rdzv-endpoint=192.168.0.35:12345 --node-rank=1 multinode.py 10 1 config.yaml
