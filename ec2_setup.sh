#!/usr/bin/env bash
# ec2_setup.sh — run once on a fresh Ubuntu 24.04 EC2 instance
# Usage:
#   scp -i your-key.pem ec2_setup.sh ubuntu@ELASTIC_IP:~
#   ssh -i your-key.pem ubuntu@ELASTIC_IP "bash ec2_setup.sh"
set -euo pipefail

REPO_DIR="$HOME/psycho-llm"
VENV_DIR="$REPO_DIR/venv"
PORT=5000
NGROK_DOMAIN="${NGROK_DOMAIN:-}"   # set via: NGROK_DOMAIN=xxx.ngrok-free.app bash ec2_setup.sh

# ── 1. System packages ─────────────────────────────────────────────────────────
echo "==> Installing system packages"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv git tmux rsync

# ── 2. ngrok ──────────────────────────────────────────────────────────────────
if ! command -v ngrok &>/dev/null; then
  echo "==> Installing ngrok"
  curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
    | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
  echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
    | sudo tee /etc/apt/sources.list.d/ngrok.list
  sudo apt-get update -qq && sudo apt-get install -y -qq ngrok
else
  echo "==> ngrok already installed"
fi

# ── 3. Repo ───────────────────────────────────────────────────────────────────
if [ ! -d "$REPO_DIR" ]; then
  echo "==> Cloning repo — edit this URL first"
  git clone https://github.com/jm-contreras/psycho-llm.git "$REPO_DIR"
else
  echo "==> Repo already present at $REPO_DIR"
fi

# ── 4. Python venv + deps ─────────────────────────────────────────────────────
echo "==> Setting up Python venv"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet flask litellm python-dotenv

# ── 5. Data directories ───────────────────────────────────────────────────────
mkdir -p "$REPO_DIR/data/raw" "$REPO_DIR/data/mturk" "$REPO_DIR/data/prolific"

# ── 6. Smoke test ─────────────────────────────────────────────────────────────
echo "==> Smoke-testing Flask app"
cd "$REPO_DIR"
python -c "from pipeline.prolific.app import create_app; create_app(); print('Flask app OK')"

# ── 7. tmux session: flask + ngrok ────────────────────────────────────────────
echo "==> Starting tmux session 'survey'"
tmux kill-session -t survey 2>/dev/null || true
tmux new-session -d -s survey -x 220 -y 50

# Window 0: Flask
tmux send-keys -t survey:0 \
  "cd $REPO_DIR && source venv/bin/activate && \
   set -a && [ -f .env ] && source .env && set +a && \
   python -m pipeline.prolific serve --port $PORT" Enter
tmux rename-window -t survey:0 flask

# Window 1: ngrok
tmux new-window -t survey -n ngrok
if [ -n "$NGROK_DOMAIN" ]; then
  tmux send-keys -t survey:ngrok \
    "ngrok http --domain=$NGROK_DOMAIN $PORT" Enter
else
  echo "NGROK_DOMAIN not set — start ngrok manually:"
  echo "  tmux attach -t survey && Ctrl-B 1"
  echo "  ngrok http --domain=YOUR-NAME.ngrok-free.app $PORT"
fi

echo ""
echo "==> Done. Attach with: tmux attach -t survey"
echo "    Window 0 (flask): Flask server"
echo "    Window 1 (ngrok): ngrok tunnel"
echo ""
echo "==> Next steps:"
echo "    1. ngrok config add-authtoken YOUR_TOKEN  (if not already done)"
echo "    2. Copy .env with PROLIFIC_FLASK_SECRET set"
echo "    3. rsync data files from your Mac:"
echo "       rsync -avz data/raw/responses.db data/mturk/ ubuntu@ELASTIC_IP:~/psycho-llm/data/"
