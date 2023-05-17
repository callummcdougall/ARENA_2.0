# Call this script with: ./start-one-lambdalab.sh <instance-id>
# Check number of arguments
if [ $# -lt 1 ]; then
    echo "./start-one-lambdalab.sh <instance-id>"
    exit 1
fi

echo '{
  "region_name": "us-south-1",
  "instance_type_name": "gpu_1x_a100",
  "ssh_key_names": [
    "wmlb-everyone"
  ],
  "file_system_names": [],
  "quantity": 1,
  "name": "wmlb-node-'$1'"
}' | curl -H "Content-Type: application/json" -H 'User-Agent: '$1 -X POST --data-binary @- -u `cat ~/Downloads/lambda_cloud_api_lambdalabs-secret.txt`: https://cloud.lambdalabs.com/api/v1/instance-operations/launch
