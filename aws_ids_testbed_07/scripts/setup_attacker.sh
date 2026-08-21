#!/usr/bin/env bash

# setup_attacker.sh
# This script prepares the attacker EC2 instance.
# The attacker generates benign traffic and controlled lab attack traffic.

set -euo pipefail

# Avoid interactive installation questions during automated SSH setup.
export DEBIAN_FRONTEND=noninteractive

# Automatically accept service restart decisions from needrestart.
# This prevents setup from hanging at "Restarting services...".
export NEEDRESTART_MODE=a

echo "[attacker] Repairing unfinished package configuration if needed..."
sudo -E dpkg --configure -a

echo "[attacker] Updating package list..."
sudo -E apt-get update

echo "[attacker] Installing traffic tools..."
sudo -E apt-get install -y \
    curl \
    apache2-utils \
    hping3 \
    nmap \
    python3 \
    python3-pip

echo "[attacker] Creating IDS lab folders..."
sudo mkdir -p /opt/aws_ids_testbed/logs

# Give the ubuntu user permission to write lab files.
sudo chown -R ubuntu:ubuntu /opt/aws_ids_testbed

echo "[attacker] Setup completed successfully."
echo "[attacker] Later benign test:"
echo "[attacker] curl http://VICTIM_PRIVATE_IP"
echo "[attacker] Later controlled HTTP flood test:"
echo "[attacker] ab -n 100 -c 5 http://VICTIM_PRIVATE_IP/"
echo "[attacker] Later controlled SYN flood test:"
echo "[attacker] sudo hping3 -S -p 80 -c 1000 VICTIM_PRIVATE_IP"
