set -e

sudo setfacl --modify user:paperspace:rw /var/run/docker.sock

# Add the ssh key to the ssh-agent
chmod 600 ~/.ssh/arena_ssh
eval "`ssh-agent -s`"
ssh-add ~/.ssh/arena_ssh
ssh-keyscan -H github.com >> ~/.ssh/known_hosts

# Log in to docker
#docker login --username=wmlbredwood --password=dckr_pat_P-ma_kUDCWwDka8W7K-ta8iGbUE
git clone git@github.com:pranavgade20/ARENA_2.0.git || cd ARENA_2.0 && git pull && cd ..
cd ARENA_2.0 
DOCKER_BUILDKIT=1 docker build --ssh default -t arena --platform linux/amd64 -f infrastructure/Dockerfile .
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
      docker run --name "group-$i" -d --runtime=nvidia --gpus '"device='"$i"'"' -p $((2222+i)):22 arena;
      sudo ufw allow $((2222+i));
    done # launch docker things
fi
