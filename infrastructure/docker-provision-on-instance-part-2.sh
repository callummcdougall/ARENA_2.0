set -e

sudo setfacl --modify user:ubuntu:rw /var/run/docker.sock

# Add the ssh key to the ssh-agent
chmod 600 ~/.ssh/arena_ssh
eval "`ssh-agent -s`"
ssh-add ~/.ssh/arena_ssh
ssh-keyscan -H github.com >> ~/.ssh/known_hosts

# Stop any existing containers
if [ -n "$(docker ps -a -q)" ]; then
    docker kill $(docker ps -a -q)
    sleep 2
fi

docker run --name "arena" -d --runtime=nvidia --gpus '"device='"0"'"' -p 2222:22 ghcr.io/callummcdougall/arena:latest;
sudo ufw allow 2222;
