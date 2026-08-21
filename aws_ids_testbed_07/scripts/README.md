# Setup Scripts

These bash scripts prepare each EC2 instance after it is created.

## setup_victim.sh

Runs on the victim EC2 instance.

It installs:

- Nginx web server
- tcpdump
- tshark
- Python basics

It also creates folders for PCAP, CSV, and logs.

## setup_attacker.sh

Runs on the attacker EC2 instance.

It installs:

- curl
- apache2-utils for `ab`
- hping3
- nmap
- Python basics

This machine will generate benign traffic and controlled lab attack traffic.

## setup_ids.sh

Runs on the IDS EC2 instance.

It installs:

- Python
- Python virtual environment tools
- tcpdump
- tshark

It also creates IDS folders for input files, output files, model files, and logs.
