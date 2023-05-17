set -e

WMLB_IMAGE=wmlb

sudo setfacl --modify user:ubuntu:rw /var/run/docker.sock

# Add the wmlb_ssh key to the ssh-agent
chmod 600 ~/.ssh/wmlb_ssh
eval "`ssh-agent -s`"
ssh-add ~/.ssh/wmlb_ssh
ssh-keyscan -H github.com >> ~/.ssh/known_hosts

# Log in to docker
#docker login --username=wmlbredwood --password=dckr_pat_P-ma_kUDCWwDka8W7K-ta8iGbUE
git clone git@github.com:pranavgade20/wmlb.git || cd wmlb && git pull && cd ..
cd wmlb
DOCKER_BUILDKIT=1 docker build --no-cache --ssh default -t wmlb --platform linux/amd64 -f infrastructure/Dockerfile .

# Stop any existing containers
if [ -n "$(docker ps -a -q)" ]; then
    docker kill $(docker ps -a -q)
    sleep 2
fi

# Start one container if there are < 8 GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# Start two containers if there are >= 8 GPUs
if [ $NUM_GPUS -ge 1 ]; then
    MEM_MB_EACH=$(expr $(free -m | grep -oP '\d+' | sed '6!d') / 8)
    CPUS_EACH=$(expr $(nproc --all) / 8)
    for i in {0..7}; do
      echo -n "$(hostname):$((2222+i))" > "$i.txt"
      docker run --name "group-$i" --rm -d --runtime=nvidia --gpus '"device='$i'"' -p $((2222+i)):22 wmlb;
      sudo ufw allow $((2222+i));
    done # launch docker things
fi
