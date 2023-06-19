# Call this script with: ./reset-repo.sh <lambda instance ip>

# Set the repo to the main branch and delete all working tree changes
ssh ubuntu@$1 'eval `ssh-agent -s`; ssh-add ~/.ssh/wmlb_ssh; cd ~/wmlb; git checkout main; git fetch origin; git reset --hard origin/main; git clean -f'

# Also reset git configs to use wmlb-account
ssh ubuntu@$1 'cd ~/wmlb; git config --unset user.name; git config --unset user.email; echo "[user]
        name = WMLB Account
        email = 110868426+wmlb-account@users.noreply.github.com
" > ~/.gitconfig'

# Also logout wandb
ssh ubuntu@$1 "sed -i '/machine api.wandb.ai/,+2d' ~/.netrc"