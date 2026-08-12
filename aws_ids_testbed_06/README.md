# AWS IDS Testbed 06

Simple AWS IDS testbed.

Goal:

```text
create EC2 instances
install victim/attacker/IDS software
generate benign and controlled attack traffic
capture PCAP
convert PCAP to CSV
run trained IDS model
show alert
```

First command:

```bash
cd /usagers3/sanazb/Projects/AWS_IDS_TestBed/aws_ids_testbed_06
python3 -m aws_ids_testbed_06.cli status
```

Current status:

```text
No AWS resources are created yet.
No traffic is generated yet.
```
