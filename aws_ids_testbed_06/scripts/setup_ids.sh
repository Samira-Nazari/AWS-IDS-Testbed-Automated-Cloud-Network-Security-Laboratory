#!/usr/bin/env bash

# setup_ids.sh
# This script prepares the IDS EC2 instance.
# The IDS receives traffic files and runs the trained IDS model later.

set -euo pipefail

# Avoid interactive installation questions during automated SSH setup.
export DEBIAN_FRONTEND=noninteractive

# Automatically accept service restart decisions from needrestart.
# This prevents setup from hanging at "Restarting services...".
export NEEDRESTART_MODE=a

echo "[ids] Repairing unfinished package configuration if needed..."
sudo -E dpkg --configure -a

echo "[ids] Updating package list..."
sudo -E apt-get update

echo "[ids] Pre-answering Wireshark/tshark installation question..."
echo "wireshark-common wireshark-common/install-setuid boolean false" | sudo debconf-set-selections

echo "[ids] Installing Python and packet analysis tools..."
sudo -E apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    tcpdump \
    tshark \
    curl

echo "[ids] Creating IDS lab folders..."
mkdir -p "$HOME/aws_ids_testbed/input"
mkdir -p "$HOME/aws_ids_testbed/output"
mkdir -p "$HOME/aws_ids_testbed/models"
mkdir -p "$HOME/aws_ids_testbed/logs"

echo "[ids] Creating Python virtual environment..."
python3 -m venv "$HOME/aws_ids_testbed/ids_env"

echo "[ids] Installing first Python libraries..."
"$HOME/aws_ids_testbed/ids_env/bin/python" -m pip install --upgrade pip
"$HOME/aws_ids_testbed/ids_env/bin/python" -m pip install numpy pandas scikit-learn joblib pyyaml fastapi uvicorn python-multipart

echo "[ids] Setup completed successfully."
echo "[ids] Later model files will go in:"
echo "[ids] $HOME/aws_ids_testbed/models"
