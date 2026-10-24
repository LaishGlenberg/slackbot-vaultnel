# Deploying Vaultnel

One always-on e2-micro VM runs the container; Cloud Build builds and pushes the
image; the VM polls `:latest` every five minutes and restarts itself when the
image changes. No inbound ports are needed for Slack — Socket Mode is outbound
only; SSH is the only thing exposed.

```
GitHub push → Cloud Build → Artifact Registry :latest
                                    ↓ (poll every 5 min)
                              VM pulls + restarts
```

## One-time GCP setup

The image is pushed into the **existing** `gcloud-backend` Artifact Registry
repository in `us-west2`, so there is nothing to create. The VM stays in
`us-central1` (`us-central1-c`) because that is an e2-micro free-tier region —
`us-west2` is not. The VM pulls the image across regions; only a real update
transfers layers, so the egress is fractions of a cent per deploy.

If you would rather keep the image in a dedicated repository, create one and
set `_REPO` in `cloudbuild.yaml` (or on the trigger) to match:

```bash
PROJECT=vaultnel
REGION=us-west2
gcloud config set project "$PROJECT"

gcloud artifacts repositories create vaultnel \
  --repository-format=docker \
  --location="$REGION" \
  --description="Vaultnel bot images"
```

No IAM binding is needed in the same project: Cloud Build can push by default,
and the VM's default service account (Editor + the default storage read-only
scope) can pull. Only add `roles/artifactregistry.writer` /
`roles/artifactregistry.reader` if the push or pull is denied.

Then create a Cloud Build trigger: **Cloud Build → Triggers → Connect
repository**, pick the GitHub repo, branch `^main$`, config file
`cloudbuild.yaml`. An existing GitHub connection in the project can be reused —
only the trigger is new, and no substitution overrides are required.

## One-time VM setup

SSH in (`gcloud compute ssh slackbot-e2-micro --zone=us-central1-c`), then:

```bash
sudo apt-get update
sudo apt-get install -y docker.io curl git
sudo systemctl enable --now docker

# grab the compose-free deploy files from the repo
git clone <YOUR_REPO_URL> ~/slackbot-vaultnel
sudo mkdir -p /opt/vaultnel
sudo cp ~/slackbot-vaultnel/.env.example /opt/vaultnel/.env
sudo cp ~/slackbot-vaultnel/deploy/update.sh \
        ~/slackbot-vaultnel/deploy/vaultnel-update.service \
        ~/slackbot-vaultnel/deploy/vaultnel-update.timer \
        /opt/vaultnel/
sudo nano /opt/vaultnel/.env   # fill tokens + LLM_API_KEY
```

Optional integrations: `mcp.json` lives on the data volume next to the
database, so put it there once (the token stays in `.env`):

```bash
sudo nano /opt/vaultnel/mcp.json      # copy mcp.example.json and edit it
sudo docker run --rm -v vaultnel-data:/data -v /opt/vaultnel:/cfg \
  alpine cp /cfg/mcp.json /data/mcp.json
```

Append the image reference to the same file:

```bash
echo 'VAULTNEL_IMAGE=us-west2-docker.pkg.dev/vaultnel/gcloud-backend/vaultnel:latest' \
  | sudo tee -a /opt/vaultnel/.env
```

Install the timer and start it:

```bash
sudo chmod +x /opt/vaultnel/update.sh
sudo cp /opt/vaultnel/vaultnel-update.service /etc/systemd/system/
sudo cp /opt/vaultnel/vaultnel-update.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now vaultnel-update.timer

sudo /opt/vaultnel/update.sh   # first pull, starts the container
sudo docker logs -f vaultnel
```

A healthy start logs `connecting to Slack over Socket Mode`, the Bolt banner,
and `heartbeat: process alive` every five minutes.

## Day to day

- **Deploy a change**: push to `main`. Cloud Build tags the commit and moves
  `:latest`; the VM updates within five minutes.
- **Deploy now**: `sudo /opt/vaultnel/update.sh`.
- **Logs**: `sudo docker logs --since 1h vaultnel`.
- **Restart**: `sudo systemctl restart docker` or let `update.sh` recreate it.
- **Rollback**: pin `VAULTNEL_IMAGE` in `/opt/vaultnel/.env` to an older
  `:<commit-sha>` tag, run `update.sh`, then switch back to `:latest` when ready.
  The tag is re-pulled from the registry, so no local copy is needed.
- **Disk**: `update.sh` keeps exactly one image (prunes before and after the
  pull) and logs `disk: …` each time. Old `:$COMMIT_SHA` tags accumulate in the
  registry, not on the VM; cap them with a cleanup policy:

  ```bash
  gcloud artifacts repositories set-cleanup-policies vaultnel --location=us-west2 \
    --policy=keep-minimum-versions=10
  ```
- **Database**: the `vaultnel-data` Docker volume holds
  `/data/vaultnel.db`; it survives image updates. Snapshot the disk before
  schema changes you do not trust.

## VM sizing notes

`e2-micro` (1 GB) is enough for the bot itself. Builds happen in Cloud Build, so
the VM never compiles anything. One image plus Docker fits comfortably in the
10 GB boot disk, but resizing is online and cheap if you ever need it:

```bash
gcloud compute disks resize slackbot-e2-micro --size=20GB --zone=us-central1-c
# then on the VM: sudo growpart /dev/sda 1 && sudo resize2fs /dev/sda1
```

Add a swap file if you plan to run other things on the box:

```bash
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```
