#!/usr/bin/env bash

# setup_victim.sh
# This script prepares the victim EC2 instance.
# The victim runs a web server and captures network traffic.

set -euo pipefail

# Avoid interactive installation questions during automated SSH setup.
export DEBIAN_FRONTEND=noninteractive

# Automatically accept service restart decisions from needrestart.
# This prevents setup from hanging at "Restarting services...".
export NEEDRESTART_MODE=a

echo "[victim] Repairing unfinished package configuration if needed..."
sudo -E dpkg --configure -a

echo "[victim] Updating package list..."
sudo -E apt-get update

echo "[victim] Pre-answering Wireshark/tshark installation question..."
echo "wireshark-common wireshark-common/install-setuid boolean false" | sudo debconf-set-selections

echo "[victim] Installing Nginx and packet capture tools..."
sudo -E apt-get install -y \
    nginx \
    tcpdump \
    tshark \
    curl \
    python3 \
    python3-pip

echo "[victim] Creating IDS lab folders..."
sudo mkdir -p /opt/aws_ids_testbed/bin
sudo mkdir -p /opt/aws_ids_testbed/config
sudo mkdir -p /opt/aws_ids_testbed/csv
sudo mkdir -p /opt/aws_ids_testbed/logs
sudo mkdir -p /opt/aws_ids_testbed/pcap/writing
sudo mkdir -p /opt/aws_ids_testbed/pcap/pending
sudo mkdir -p /opt/aws_ids_testbed/pcap/sent
sudo mkdir -p /opt/aws_ids_testbed/pcap/failed

# Give the ubuntu user permission to write lab files.
sudo chown -R ubuntu:ubuntu /opt/aws_ids_testbed

echo "[victim] Creating a simple test web page..."
sudo tee /var/www/html/index.html >/dev/null <<'HTML'
<!doctype html>
<html>
  <head>
    <title>AWS IDS Testbed Victim</title>
  </head>
  <body>
    <h1>AWS IDS Testbed Victim</h1>
    <p>Nginx is running on the victim EC2 instance.</p>
  </body>
</html>
HTML

echo "[victim] Enabling and starting Nginx..."
sudo systemctl enable nginx
sudo systemctl restart nginx

echo "[victim] Setup completed successfully."
echo "[victim] Later test from attacker with:"
echo "[victim] curl http://VICTIM_PRIVATE_IP"
