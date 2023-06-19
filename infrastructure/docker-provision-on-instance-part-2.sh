set -e

sudo setfacl --modify user:ubuntu:rw /var/run/docker.sock

# Add the ssh key to the ssh-agent
# Stop any existing containers
if [ -n "$(docker ps -a -q)" ]; then
    docker kill $(docker ps -a -q)
    sleep 2
fi

docker run --name "arena" -d --runtime=nvidia --gpus '"device='"0"'"' -p 2222:22 ghcr.io/pranavgade20/arena:latest;
sudo ufw allow 2222;
