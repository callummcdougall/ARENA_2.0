#!/bin/bash

cd
# check if imagenet_38k.zip exists
if [ ! -f imagenet_38k.zip ]; then
    wget http://192.9.135.130:8000/ILSVRC/Data/CLS-LOC/imagenet_38k.zip -O imagenet_38k.zip
    unzip imagenet_38k.zip
fi

export NCCL_DEBUG=TRACE
export NCCL_P2P_DISABLE=1
#export NCCL_P2P_LEVEL=LOC
export NCCL_SHM_DISABLE=1
#export NCCL_SOCKET_IFNAME="=tailscale0"
##export NCCL_DEBUG_SUBSYS=COLL
#export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=1
export PYTHONUNBUFFERED=1
export ACCELERATE_DISABLE_RICH=1
export LD_PRELOAD=/home/ubuntu/libnccl.so.2.18.1


ps aux | grep "foo" | grep -v grep | awk '{print $2}' | grep -v $$ | xargs kill -9  # clean up previous processes
sudo ufw allow 12345
echo 'Running python '"$1"' '"${*:2}" > log.txt 2>&1 &
nohup python $1 "${@:2}" > log.txt 2>&1 &
#nohup python -m trace --ignore-dir /usr:/home/ubuntu/.local/lib -t $1 "${*:2}" > log.txt 2>&1 &

