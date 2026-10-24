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

Set the same region here, in `cloudbuild.yaml` (`_REGION`) and when creating the
repository. The defaults assume `us-west1`, a free-tier region for e2-micro.

```bash
PROJECT=vaultnel
REGION=us-west1
gcloud config set project "$PROJECT"

gcloud artifacts repositories create vaultnel \
  --repository-format=docker \
  --location="$REGION" \
  --description="Vaultnel bot images"

# Cloud Build pushes images
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')@cloudbuild.gserviceaccount.com" \
  --role=roles/artifactregistry.writer

# The VM pulls images
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:$(gcloud compute instances describe slackbot-e2-micro --zone=us-west1-b --format='value(serviceAccounts[0].email)')" \
  --role=roles/artifactregistry.reader
```

Then create a Cloud Build trigger: **Cloud Build → Triggers → Connect
repository**, pick the GitHub repo, branch `^main$`, config file
`cloudbuild.yaml`. If the VM is not in `us-west1`/`us-west1-b`, override
`_REGION` in the trigger settings (and adjust the zone below).

## One-time VM setup

SSH in (`gcloud compute ssh slackbot-e2-micro --zone=us-west1-b`), then:

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

Append the image reference to the same file:

```bash
echo 'VAULTNEL_IMAGE=us-west1-docker.pkg.dev/vaultnel/vaultnel/vaultnel:latest' \
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
- **Database**: the `vaultnel-data` Docker volume holds
  `/data/vaultnel.db`; it survives image updates. Snapshot the disk before
  schema changes you do not trust.

## VM sizing notes

`e2-micro` (1 GB) is enough for the bot itself. Builds happen in Cloud Build, so
the VM never compiles anything. Add a swap file if you plan to run other things
on the box:

```bash
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```
