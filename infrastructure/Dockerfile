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
# for i in {0..7}; do docker run --rm -d --runtime=nvidia --gpus '"device='$i'"' -p $((2222+i)):22 wmlb; done # launch docker things

FROM nvcr.io/nvidia/pytorch:23.04-py3

RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install wget git -y

# Install dependencies for w3d2
RUN DEBIAN_FRONTEND=noninteractive TZ=America/Los_Angeles apt-get install swig cmake freeglut3-dev xvfb -y

# Clone the repo
RUN mkdir -p -m 0700 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
RUN --mount=type=ssh cd $HOME && git clone git@github.com:pranavgade20/ARENA_2.0.git

# Install the repo dependencies
ARG WMLB_PATH=/workspace/arena
ARG ENV_PATH=/root/wmlb/.env/

# Set up pre-commit hook to protect main branch
RUN cp $WMLB_PATH/infrastructure/pre-commit $WMLB_PATH/.git/hooks/pre-commit

# Install pycuda (needed for w1d6) and ffmpeg (needed for w3d2)

# Install VSCode extensions if they aren't there
RUN echo '\n\
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
RUN echo 'chmod 600 ~/.ssh/arena_ssh\n\
eval "`ssh-agent -s`"\n\
ssh-add ~/.ssh/arena_ssh' >> ~/.bashrc

# Set up .gitconfig
RUN echo '[user]\n\
        name = ARENA Account\n\
        email = pranav@arena.education\n\
' > ~/.gitconfig

# Skip over prompt to add github to known_hosts
RUN ssh-keyscan -H github.com >> ~/.ssh/known_hosts

RUN apt-get install openssh-server -y
RUN echo 'Include /etc/ssh/sshd_config.d/*.conf\n\
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
RUN echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAJZyTBrpcH1u4WHGKNCgIKV1V+S8xHsX4Z+OWd4c0Zq pranavgade20+wmlb@gmail.com' > /etc/ssh/authorized_keys
COPY infrastructure/arena_ssh /root/.ssh
COPY infrastructure/arena_ssh.pub /root/.ssh
COPY infrastructure/ssh /etc/ssh
RUN echo "export ACCELERATE_DISABLE_RICH=1" >> /etc/profile

WORKDIR $WMLB_PATH

CMD /usr/sbin/sshd -D
