#!/usr/bin/env bash
# Bootstrap script for EC2-benchmark (Ubuntu 22.04+).
# Idempotent: safe to re-run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log() { printf "\033[1;36m[setup]\033[0m %s\n" "$*"; }

log "Updating apt package index"
sudo apt-get update -y

log "Installing system packages (Python 3.11+, build tools, curl, git)"
sudo apt-get install -y \
    python3 \
    python3-venv \
    python3-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    ca-certificates

if ! command -v docker >/dev/null 2>&1; then
    log "Installing Docker via convenience script"
    curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
    sudo sh /tmp/get-docker.sh
    rm -f /tmp/get-docker.sh
else
    log "Docker already installed: $(docker --version)"
fi

if ! groups "$USER" | grep -q '\bdocker\b'; then
    log "Adding $USER to the 'docker' group (re-login required to take effect)"
    sudo usermod -aG docker "$USER"
fi

PYTHON_BIN="python3"
PY_MAJ_MIN=$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
log "Detected Python $PY_MAJ_MIN"

if [ ! -d ".venv" ]; then
    log "Creating virtualenv at .venv"
    $PYTHON_BIN -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

log "Upgrading pip and installing the package in editable mode"
pip install --upgrade pip
pip install -e ".[dev]"

cat <<'EOF'

============================================================
 masops-evaluation: EC2-benchmark setup complete
============================================================

Next steps:

  1. Copy the env template and fill in the values:
       cp .env.example .env
       # then edit .env and set at least MASOPS_HOST

  2. Activate the virtualenv in new shells:
       source .venv/bin/activate

  3. Verify Docker permissions (you may need to log out and back
     in after the usermod above):
       docker ps

  4. Select the SWE-bench instances for your rodada:
       select-instances --n-per-difficulty 15 --seed 42

  5. Run an evaluation rodada:
       run-evaluation --rodada-id v1 --repetitions 3

  6. Aggregate results into a report:
       aggregate-results --rodada-id v1

EOF
