# Call this script with: ./deploy.sh <cluster id> <lambda instance ip> [optional: kwargs]
# <host> <run-on-server args>

# Check number of arguments
if [ $# -lt 2 ]; then
    echo "./deploy.sh <cluster id> <lambda instance ip> [optional: kwargs]"
    exit 1
fi

# TODO parse args
scp data_parallel_inference.py arena2

ssh ubuntu@$2 'bash -s' < run-on-server.sh "$1" "${@:3}"


