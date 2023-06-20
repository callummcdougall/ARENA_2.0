# Call this script with: ./deploy.sh <cluster id> <lambda instance ip> [optional: kwargs]

# Check number of arguments
if [ $# -lt 2 ]; then
    echo "./deploy.sh <cluster id> <lambda instance ip> [optional: kwargs]"
    exit 1
fi

scp pipeline_parallel.py ubuntu@$2:~/

ssh ubuntu@$2 'bash -s' < run-on-server.sh "$1" "${@:3}"


