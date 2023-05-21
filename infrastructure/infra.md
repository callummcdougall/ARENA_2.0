# To provision a single instance
1. Make sure you have the `wmlb_ssh` key at `~/.ssh/wmlb_ssh` on your machine
2. Create a new instance in Lambda Labs, wait for it to boot up, and get its IP address
3. In this directory, run the following: `./provision.sh <lambda instance ip>`

# To provision multiple instances
1. Make sure you have the `wmlb_ssh` key at `~/.ssh/wmlb_ssh` on your machine
2. Make sure you have `parallel` installed (on Mac, install with `brew install parallel`)
3. Create a number of instances in Lambda Labs and wait for them to boot up
4. Create a file in this directory that contains their IP addresses as a newline-delimited list:
    ```
    <instance ip 1>
    <instance ip 2>
    <instance ip 3>
    ...
    <instance ip n>
    ```
5. In this directory, run the following: `parallel -a <ip addresses list file> ./provision.sh`

Provisioning should take 10-20 minutes.

# To reset the state of the repo

This command sets the repo to the main branch and deletes all working tree changes.

For a single instance: `./reset-repo.sh <instance ip>`

For multiple instances: `parallel -a <ip addresses list file> ./reset-repo.sh` 