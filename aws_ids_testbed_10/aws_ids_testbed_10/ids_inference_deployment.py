"""Define the files required by the AWS IDS inference runtime."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from aws_ids_testbed_10.remote_runner import RemoteRunner
from aws_ids_testbed_10.remote_settings import (
    get_private_key_path,
    get_public_host,
    get_ssh_username,
)


REMOTE_BASE_DIRECTORY = PurePosixPath(
    "/home/ubuntu/aws_ids_testbed"
)


@dataclass(frozen=True)
class RuntimeDeploymentFile:
    """Map one local runtime source to its AWS destination."""

    source_path: Path
    remote_path: PurePosixPath


_INFERENCE_MODULES = (
    "__init__.py",
    "ids_input_agent.py",
    "ids_inference_agent.py",
    "ids_inference_input.py",
    "ids_inference_pipeline.py",
    "ids_inference_queue.py",
    "ids_inference_scenario.py",
    "ids_inference_transaction.py",
    "ids_inference_workspace.py",
    "ids_master_predictions.py",
    "ids_scenario_lifecycle.py",
    "ids_stream_buffer_store.py",
    "ids_stream_continuity.py",
)


_MODEL_ARTIFACTS = (
    "Lite_V9_CICIoT2023/outputs/models/best_model.pt",
    "Lite_V9_CICIoT2023/outputs/models/model_config.json",
    "Lite_V9_CICIoT2023/outputs/data/selected_features.pkl",
    "Lite_V9_CICIoT2023/outputs/data/standard_scaler.pkl",
    "Lite_V9_CICIoT2023/outputs/data/window_config.pkl",
    "Lite_V9_CICIoT2023/outputs/data/label_classes.pkl",
)


def build_runtime_deployment_manifest(
    project_root: Path,
) -> tuple[RuntimeDeploymentFile, ...]:
    """Return and validate the exact inference runtime files."""

    project_root = Path(project_root)
    deployment_files: list[RuntimeDeploymentFile] = []

    def add_file(
        source_relative_path: str,
        remote_relative_path: str | None = None,
    ) -> None:
        source_path = project_root / source_relative_path
        destination = (
            remote_relative_path or source_relative_path
        )

        if not source_path.is_file():
            raise FileNotFoundError(
                f"Required runtime file is missing: {source_path}"
            )

        if source_path.stat().st_size == 0:
            raise ValueError(
                f"Required runtime file is empty: {source_path}"
            )

        deployment_files.append(
            RuntimeDeploymentFile(
                source_path=source_path,
                remote_path=(
                    REMOTE_BASE_DIRECTORY / destination
                ),
            )
        )

    add_file(
        "requirements-ids-runtime.txt",
    )
    add_file(
        "scripts/setup_ids_inference_runtime.sh",
        "bin/setup_ids_inference_runtime.sh",
    )

    for module_name in _INFERENCE_MODULES:
        add_file(
            f"aws_ids_testbed_10/{module_name}"
        )

    detector_directory = (
        project_root / "ids_detector" / "lite_v6"
    )

    for source_path in sorted(
        detector_directory.rglob("*.py")
    ):
        relative_path = source_path.relative_to(project_root)

        add_file(str(relative_path))

    for artifact_path in _MODEL_ARTIFACTS:
        add_file(artifact_path)

    return tuple(deployment_files)


def _run_remote_command(
    ssh_client,
    command: str,
) -> None:
    """Run one required deployment command."""

    _stdin, stdout, stderr = ssh_client.exec_command(
        command,
        get_pty=True,
    )

    for line in stdout:
        print(line, end="")

    exit_code = stdout.channel.recv_exit_status()

    if exit_code != 0:
        error_text = stderr.read().decode(
            "utf-8",
            errors="replace",
        )
        raise RuntimeError(
            f"Remote deployment command failed: {error_text}"
        )


def upload_ids_inference_runtime(
    project_root: Path,
) -> int:
    """Upload the validated inference runtime to the IDS."""

    import paramiko

    project_root = Path(project_root)
    manifest = build_runtime_deployment_manifest(
        project_root
    )

    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    ids_host = get_public_host(project_root, "ids")

    if not private_key_path.is_file():
        raise FileNotFoundError(
            f"Private key not found: {private_key_path}"
        )

    remote_directories = sorted(
        {
            str(deployment_file.remote_path.parent)
            for deployment_file in manifest
        }
    )

    ssh_client = paramiko.SSHClient()
    ssh_client.load_system_host_keys()
    ssh_client.set_missing_host_key_policy(
        paramiko.AutoAddPolicy()
    )

    print(f"Connecting to {username}@{ids_host} ...")

    ssh_client.connect(
        hostname=ids_host,
        username=username,
        key_filename=str(private_key_path),
        timeout=30,
    )

    try:
        mkdir_command = (
            "mkdir -p "
            + " ".join(
                shlex.quote(directory)
                for directory in remote_directories
            )
        )

        print("Creating inference runtime directories...")
        _run_remote_command(
            ssh_client,
            mkdir_command,
        )

        print(
            f"Uploading {len(manifest)} inference runtime files..."
        )

        with ssh_client.open_sftp() as sftp:
            for deployment_file in manifest:
                print(
                    "[ids-inference-deploy] "
                    f"{deployment_file.source_path.name} "
                    f"→ {deployment_file.remote_path}"
                )

                sftp.put(
                    str(deployment_file.source_path),
                    str(deployment_file.remote_path),
                )

            sftp.chmod(
                str(
                    REMOTE_BASE_DIRECTORY
                    / "bin"
                    / "setup_ids_inference_runtime.sh"
                ),
                0o700,
            )

        print("IDS inference runtime files uploaded.")
        return 0

    finally:
        ssh_client.close()


def setup_uploaded_ids_inference_runtime(
    project_root: Path,
) -> int:
    """Install dependencies and validate the uploaded IDS runtime."""

    project_root = Path(project_root)

    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    ids_host = get_public_host(project_root, "ids")

    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    remote_base = str(REMOTE_BASE_DIRECTORY)
    python_bin = (
        f"{remote_base}/ids_inference_env/bin/python"
    )
    setup_script = (
        f"{remote_base}/bin/"
        "setup_ids_inference_runtime.sh"
    )

    command = (
        f"bash {setup_script} && "
        f"cd {remote_base} && "
        f"{python_bin} -m py_compile "
        f"{remote_base}/aws_ids_testbed_10/"
        "ids_input_agent.py "
        f"{remote_base}/aws_ids_testbed_10/"
        "ids_inference_agent.py && "
        f"{python_bin} -c \""
        "from ids_detector.lite_v6.evaluation.evaluator "
        "import _load_lite_v6_trained_model; "
        "_load_lite_v6_trained_model('cpu'); "
        "print('Lite V9 model loaded successfully on CPU.')"
        "\""
    )

    return runner.run_command(
        host=ids_host,
        command=command,
    )


def deploy_ids_inference_runtime(
    project_root: Path,
) -> int:
    """Upload, install, and verify the AWS inference runtime."""

    upload_status = upload_ids_inference_runtime(
        project_root
    )

    if upload_status != 0:
        return upload_status

    return setup_uploaded_ids_inference_runtime(
        project_root
    )
