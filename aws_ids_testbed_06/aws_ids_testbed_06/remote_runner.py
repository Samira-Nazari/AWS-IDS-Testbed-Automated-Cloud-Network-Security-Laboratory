"""Run local bash scripts on remote EC2 instances over SSH."""

from __future__ import annotations

from pathlib import Path


class RemoteRunner:
    """Upload one local script to one EC2 instance and run it."""

    def __init__(self, username: str, private_key_path: Path) -> None:
        """Store SSH login information.

        Args:
            username: SSH username, usually ubuntu for Ubuntu EC2.
            private_key_path: Path to the .pem private key file.
        """
        self.username = username
        self.private_key_path = private_key_path

    def run_script(self, host: str, local_script_path: Path) -> int:
        """Upload and run a local bash script on a remote EC2 instance.

        Args:
            host: Public IP address or DNS name of the EC2 instance.
            local_script_path: Local path to the bash script.

        Returns:
            Remote command exit code. Zero means success.
        """
        import paramiko

        if not self.private_key_path.exists():
            raise FileNotFoundError(f"Private key not found: {self.private_key_path}")

        if not local_script_path.exists():
            raise FileNotFoundError(f"Script not found: {local_script_path}")

        remote_script_path = f"/tmp/{local_script_path.name}"

        ssh_client = paramiko.SSHClient()

        # Use known_hosts when possible.
        ssh_client.load_system_host_keys()

        # In this private lab, automatically accept a new EC2 host key.
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        print(f"Connecting to {self.username}@{host} ...")
        ssh_client.connect(
            hostname=host,
            username=self.username,
            key_filename=str(self.private_key_path),
            timeout=30,
        )

        try:
            print(f"Uploading {local_script_path} to {remote_script_path} ...")
            with ssh_client.open_sftp() as sftp:
                sftp.put(str(local_script_path), remote_script_path)
                sftp.chmod(remote_script_path, 0o700)

            print(f"Running {remote_script_path} ...")
            _stdin, stdout, _stderr = ssh_client.exec_command(
                f"bash {remote_script_path}",
                get_pty=True,
            )

            for line in stdout:
                print(line, end="")

            exit_code = stdout.channel.recv_exit_status()
            print(f"Remote script finished with exit code: {exit_code}")
            return exit_code

        finally:
            ssh_client.close()

    def run_command(self, host: str, command: str) -> int:
        """Run one shell command on a remote EC2 instance.

        Args:
            host: Public IP address or DNS name of the EC2 instance.
            command: Linux command to run on the remote instance.

        Returns:
            Remote command exit code. Zero means success.
        """
        import paramiko

        if not self.private_key_path.exists():
            raise FileNotFoundError(f"Private key not found: {self.private_key_path}")

        ssh_client = paramiko.SSHClient()

        # Use known_hosts when possible.
        ssh_client.load_system_host_keys()

        # In this private lab, automatically accept a new EC2 host key.
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        print(f"Connecting to {self.username}@{host} ...")
        ssh_client.connect(
            hostname=host,
            username=self.username,
            key_filename=str(self.private_key_path),
            timeout=30,
        )

        try:
            print("Running remote command:")
            print(command)

            _stdin, stdout, _stderr = ssh_client.exec_command(
                command,
                get_pty=True,
            )

            for line in stdout:
                print(line, end="")

            exit_code = stdout.channel.recv_exit_status()
            print(f"Remote command finished with exit code: {exit_code}")
            return exit_code

        finally:
            ssh_client.close()
