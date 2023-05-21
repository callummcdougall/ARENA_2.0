# Call this script with: ./docker-provision.sh <lambda instance ip> [optional: path to SSH key for Git repository]

# Check number of arguments
if [ $# -lt 1 ]; then
    echo "./provision.sh <lambda instance ip> [optional: SSH key path]"
    exit 1
fi

# Default SSH key if not provided, check if file exists
SSH_KEY=${2:-~/.ssh/id_rsa}
echo "Using SSH key: $SSH_KEY"

if [ ! -f "$SSH_KEY" ]; then
    echo "SSH key does not exist at $SSH_KEY. Please ensure key is provided and re-run script."
    exit 1
fi

# Skip over the prompt to add the instance to known_hosts
ssh-keyscan -H $1 >> ~/.ssh/known_hosts

# Copy gituhb ssh key to the remote machine
scp arena_ssh ubuntu@$1:/home/ubuntu/.ssh/arena_ssh

# Install nvidia-container-toolkit and nvidia-container-runtime
ssh ubuntu@$1 'bash -s' < docker-provision-on-instance-part-1.sh 

# The above will reboot the instance; wait until it's back up to finish
ssh -q ubuntu@$1 exit
until [ $? -eq 0 ]
  do
    echo 'Waiting until the instance has restarted...'
    sleep 2
    ssh -q ubuntu@$1 exit
  done

# Pull the wmlbredwood/wmlb image and start one or two containers
# ssh ubuntu@$1 'bash -s' < docker-provision-on-instance-part-2.sh
