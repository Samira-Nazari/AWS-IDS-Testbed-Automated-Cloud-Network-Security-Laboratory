#!/usr/bin/env bash

# setup_benign_generator.sh
# Shared setup for all five benign traffic generator instances.

set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a

echo "[benign-generator] Repairing package configuration..."
sudo -E dpkg --configure -a

echo "[benign-generator] Updating package list..."
sudo -E apt-get update

echo "[benign-generator] Installing benign traffic tools..."
sudo -E apt-get install -y \
    ca-certificates \
    curl \
    dnsutils \
    iputils-ping \
    mosquitto-clients \
    netcat-openbsd \
    python3 \
    python3-pip

echo "[benign-generator] Creating directories..."
sudo mkdir -p \
    /opt/aws_ids_testbed/benign/bin \
    /opt/aws_ids_testbed/benign/config \
    /opt/aws_ids_testbed/benign/logs

sudo chown -R ubuntu:ubuntu /opt/aws_ids_testbed

echo "[benign-generator] Setup completed successfully."
