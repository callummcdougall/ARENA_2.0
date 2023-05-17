# Call as ./commit-and-push.sh <ip> <w#d#>
# Parallel call :
# parallel ./commit-and-push.sh :::: <ip addresses file> ::: <w#d#>

# Skip over the prompt to add the instance to known_hosts
ssh-keyscan -H $1 >> ~/.ssh/known_hosts

ssh ubuntu@$1 'bash -s' < commit-and-push-on-instance.sh "$1" "$2"