# Call this script with: ./provision.sh <lambda instance ip> [optional: path to SSH key for Git repository]

# Check number of arguments
if [ $# -lt 1 ]; then
    echo "./provision.sh <lambda instance ip> [optional: SSH key path]"
    exit 1
fi

# Default SSH key if not provided, check if file exists
SSH_KEY=${2:-~/.ssh/wmlb_ssh}
echo "Using SSH key: $SSH_KEY"

if [ ! -f "$SSH_KEY" ]; then
    echo "SSH key does not exist at $SSH_KEY. Please ensure key is provided and re-run script."
    exit 1
fi

# Skip over the prompt to add the instance to known_hosts
ssh-keyscan -H $1 >> ~/.ssh/known_hosts

# Copy gituhb ssh key to the remote machine
scp $SSH_KEY ubuntu@$1:/home/ubuntu/.ssh/wmlb_ssh

# Copy keys for RRFS assuming they're in the regular folder (credentials and config)
scp -r ~/.aws ubuntu@$1:/home/ubuntu/


# Do the rest of the provisioning steps via ssh
ssh ubuntu@$1 'bash -s' < run-on-instance-to-provision.sh


