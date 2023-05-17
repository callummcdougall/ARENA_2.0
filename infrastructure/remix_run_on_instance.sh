#!/bin/sh

# Differences from WMLB install: using Unity versions of dependencies, so some days won't work (but we only need w2d1)

wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh
bash Miniconda3-py39_4.12.0-Linux-x86_64.sh -b -p
rm -rf Miniconda3-py39_4.12.0-Linux-x86_64.sh
PATH=$HOME/miniconda3/bin:$PATH
conda init


if [ ! -f ~/.ssh/wmlb_ssh ]; then
    echo "~/.ssh/wmlb_ssh does not exist on remote instance. Please ensure key is provided and re-run script."
    exit 1
fi

chmod 600 ~/.ssh/wmlb_ssh
eval `ssh-agent -s`
ssh-add ~/.ssh/wmlb_ssh
ssh-keyscan -H github.com >> ~/.ssh/known_hosts

ENV_PATH=~/wmlb/.env/
git clone git@github.com:wmlb-account/wmlb.git
cd wmlb
conda create -p $ENV_PATH python=3.9 -y

# TBD: need to have key or key forwarding to successfully clone unity here
cd ..
git clone git@github.com:redwoodresearch/unity.git
cd unity

conda run -p $ENV_PATH pip install -r pre_requirements.txt
conda run -p $ENV_PATH pip install -r requirements.txt
conda run -p $ENV_PATH pip install -r requirements_python_rust.txt
conda run -p $ENV_PATH pip install -r requirements_non_interp.txt
conda run -p $ENV_PATH pip install --editable .

# RRFS access - TBD: does this automatically mount on restart?
rm -rf goofys # in case something was already there? confused
wget "https://github.com/kahing/goofys/releases/latest/download/goofys"
chmod +x goofys
mkdir -p ~/rrfs
./goofys --region us-west-2 redwoodfs ~/rrfs


###### Set up .bashrc and .gitconfig ######

# Add the ssh key to the ssh-agent each time
echo '
# Add github service account SSH key to agent
if [ -z "$SSH_AUTH_SOCK" ] ; then
  eval `ssh-agent -s`
  ssh-add ~/.ssh/wmlb_ssh
fi
' >> ~/.bashrc

# Install VSCode extensions if they aren't there
echo '
# Install VSCode extensions if needed
if which code &> /dev/null; then 
  extensions=$(code --list-extensions)
  if ! [[ $extensions == *"ms-python.python"* && $extensions == *"bierner.markdown-mermaid"* ]]; then
    echo "Installing VSCode extensions..."
    code --install-extension ms-python.python &> /dev/null
    code --install-extension bierner.markdown-mermaid &> /dev/null
  fi
fi
' >> ~/.bashrc

# Activate the wmlb virtualenv each time
echo '
conda activate ~/wmlb/.env/
export PATH="/home/ubuntu/wmlb/.env/bin:$PATH"
' >> ~/.bashrc


# Set up .gitconfig
echo '[user]
        name = WMLB Account
        email = 110868426+wmlb-account@users.noreply.github.com
' > ~/.gitconfig
