#!/usr/bin/env bash

# setup_victim_benign_services.sh
# Configure benign IoT-like services on the victim instance.

set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a

CONFIG_FILE="/opt/aws_ids_testbed/config/benign.env"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Missing configuration: $CONFIG_FILE"
    echo "Run configure-benign-network first."
    exit 1
fi

source "$CONFIG_FILE"

: "${VICTIM_PRIVATE_IP:?Missing VICTIM_PRIVATE_IP}"
: "${VICTIM_HTTPS_PORT:?Missing VICTIM_HTTPS_PORT}"
: "${VICTIM_DNS_PORT:?Missing VICTIM_DNS_PORT}"
: "${VICTIM_MQTT_PORT:?Missing VICTIM_MQTT_PORT}"
: "${VICTIM_UDP_PORT:?Missing VICTIM_UDP_PORT}"

echo "[victim-benign] Installing service packages..."
sudo -E apt-get update
sudo -E apt-get install -y \
    dnsmasq \
    mosquitto \
    netcat-openbsd \
    openssl

echo "[victim-benign] Configuring DNS..."
sudo tee /etc/dnsmasq.d/aws_ids_testbed_benign.conf >/dev/null <<EOF
interface=ens5
listen-address=${VICTIM_PRIVATE_IP}
bind-interfaces
port=${VICTIM_DNS_PORT}
no-resolv
address=/iot.local/${VICTIM_PRIVATE_IP}
EOF

echo "[victim-benign] Configuring MQTT..."
sudo tee /etc/mosquitto/conf.d/aws_ids_testbed_benign.conf >/dev/null <<EOF
listener ${VICTIM_MQTT_PORT} 0.0.0.0
allow_anonymous true
persistence false
EOF

echo "[victim-benign] Creating HTTPS certificate..."
sudo mkdir -p /etc/ssl/aws_ids_testbed

if [[ ! -f /etc/ssl/aws_ids_testbed/victim.crt ]]; then
    sudo openssl req -x509 -nodes -days 365 \
        -newkey rsa:2048 \
        -keyout /etc/ssl/aws_ids_testbed/victim.key \
        -out /etc/ssl/aws_ids_testbed/victim.crt \
        -subj "/CN=aws-ids-victim"
fi

echo "[victim-benign] Configuring HTTPS..."
sudo tee /etc/nginx/sites-available/aws_ids_testbed_https >/dev/null <<EOF
server {
    listen ${VICTIM_HTTPS_PORT} ssl;
    server_name _;

    ssl_certificate /etc/ssl/aws_ids_testbed/victim.crt;
    ssl_certificate_key /etc/ssl/aws_ids_testbed/victim.key;

    root /var/www/html;
    index index.html;
}
EOF

sudo ln -sf \
    /etc/nginx/sites-available/aws_ids_testbed_https \
    /etc/nginx/sites-enabled/aws_ids_testbed_https

echo "[victim-benign] Creating UDP responder..."
sudo mkdir -p /opt/aws_ids_testbed/bin

sudo tee /opt/aws_ids_testbed/bin/benign_udp_responder.py >/dev/null <<'PY'
#!/usr/bin/env python3

import os
import socket

port = int(os.environ.get("BENIGN_UDP_PORT", "9999"))

server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.bind(("0.0.0.0", port))

while True:
    data, address = server.recvfrom(4096)
    server.sendto(b"telemetry-ok\n", address)
PY

sudo chmod 755 /opt/aws_ids_testbed/bin/benign_udp_responder.py

sudo tee /etc/systemd/system/aws-ids-benign-udp.service >/dev/null <<EOF
[Unit]
Description=AWS IDS benign UDP responder
After=network-online.target

[Service]
ExecStart=/usr/bin/python3 /opt/aws_ids_testbed/bin/benign_udp_responder.py
Environment=BENIGN_UDP_PORT=${VICTIM_UDP_PORT}
Restart=always
User=ubuntu

[Install]
WantedBy=multi-user.target
EOF

echo "[victim-benign] Starting services..."
sudo nginx -t
sudo systemctl daemon-reload
sudo systemctl enable dnsmasq
sudo systemctl restart dnsmasq

sudo systemctl enable mosquitto
sudo systemctl restart mosquitto

sudo systemctl enable aws-ids-benign-udp
sudo systemctl restart aws-ids-benign-udp

sudo systemctl reload nginx

echo "[victim-benign] Victim services configured successfully."
