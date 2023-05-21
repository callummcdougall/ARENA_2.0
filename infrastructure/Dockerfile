# To run this image locally, run these commands and then use the VSCode remote extension to
# "Attach to running container...":

# Or you can get jsut a terminal in the container with this command:

# docker exec -it wmlb bash

# And then to shut it down when you're done:

# docker kill wmlb

# ------------------------------------------------------------------------------------------

# You should be able to build the container if you have ssh access to the wmlb repo

# DOCKER_BUILDKIT=1 docker build --ssh default -t wmlb --platform linux/amd64 -f infrastructure/Dockerfile .

# ------------------------------------------------------------------------------------------

# to launch multiple instances on the server:
# for i in {0..7}; do docker run --rm -d --runtime=nvidia --gpus '"device='$i'"' -p $((2222+i)):22 arena; done # launch docker things
FROM nvcr.io/nvidia/pytorch:23.04-py3

RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install wget git -y

# Install dependencies for w3d2
# RUN DEBIAN_FRONTEND=noninteractive TZ=America/Los_Angeles apt-get install swig cmake freeglut3-dev xvfb -y
RUN DEBIAN_FRONTEND=noninteractive TZ=America/Los_Angeles apt-get install openssh-server -y

# Clone the repo
RUN mkdir -p -m 0700 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

COPY infrastructure/arena_ssh /root/.ssh/
RUN pwd && find ~/.ssh && chmod 0600 ~/.ssh/arena_ssh  # just in case perms were messed up
COPY infrastructure/arena_ssh.pub /root/.ssh/
RUN cd $HOME && GIT_SSH_COMMAND='ssh -i /root/.ssh/arena_ssh -o IdentitiesOnly=yes' git clone git@github.com:callummcdougall/ARENA_2.0.git

# Install the repo dependencies
ARG ENV_PATH=~/ARENA_2.0/.env/

RUN pip install -r ~/ARENA_2.0/requirements.txt

# Install VSCode extensions if they aren't there
RUN echo -ne '\n\
if which code &> /dev/null; then \n\
  extensions=$(code --list-extensions) \n\
  if ! [[ $extensions == *"ms-python.python"* && $extensions == *"bierner.markdown-mermaid"* ]]; then \n\
    echo "Installing VSCode extensions..." \n\
    code --install-extension ms-python.python &> /dev/null \n\
    code --install-extension bierner.markdown-mermaid &> /dev/null \n\
  fi \n\
fi \n\
' >> ~/.bashrc

# Add the key to ssh-agent
RUN echo -ne 'chmod 600 ~/.ssh/arena_ssh\n\
eval "`ssh-agent -s`"\n\
ssh-add ~/.ssh/arena_ssh' >> ~/.bashrc

# Set up .gitconfig
RUN echo -ne '[user]\n\
        name = ARENA Account\n\
        email = pranav@arena.education\n\
' > ~/.gitconfig

# Skip over prompt to add github to known_hosts
RUN ssh-keyscan -H github.com >> ~/.ssh/known_hosts

RUN echo -ne 'Include /etc/ssh/sshd_config.d/*.conf\n\
PermitRootLogin yes\n\
AuthorizedKeysFile  /etc/ssh/authorized_keys\n\
PubkeyAuthentication yes\n\
PasswordAuthentication no\n\
ChallengeResponseAuthentication no\n\
UsePAM yes\n\
X11Forwarding yes\n\
PrintMotd no\n\
AcceptEnv LANG LC_*\n\
Subsystem       sftp    /usr/lib/openssh/sftp-server' > /etc/ssh/sshd_config
RUN mkdir /run/sshd
RUN echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAJZyTBrpcH1u4WHGKNCgIKV1V+S8xHsX4Z+OWd4c0Zq pranavgade20+arena@gmail.com' > /etc/ssh/authorized_keys
COPY infrastructure/ssh /etc/ssh
RUN chmod 600 /etc/ssh/*key
RUN echo "export ACCELERATE_DISABLE_RICH=1" >> /etc/profile

WORKDIR /root/

CMD /usr/sbin/sshd -D

LABEL org.opencontainers.image.source="https://github.com/callummcdougall/ARENA_2.0"
