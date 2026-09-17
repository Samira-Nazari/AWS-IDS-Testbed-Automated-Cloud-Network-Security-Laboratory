import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from aws_ids_testbed_10.ids_inference_deployment import (
    REMOTE_BASE_DIRECTORY,
    RuntimeDeploymentFile,
    build_runtime_deployment_manifest,
    deploy_ids_inference_runtime,
    setup_uploaded_ids_inference_runtime,
    upload_ids_inference_runtime,
)


class TestIdsInferenceDeployment(unittest.TestCase):
    def test_manifest_contains_only_required_runtime_files(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        manifest = build_runtime_deployment_manifest(project_root)

        remote_paths = {
            str(deployment_file.remote_path)
            for deployment_file in manifest
        }

        expected_paths = {
            str(
                REMOTE_BASE_DIRECTORY
                / "requirements-ids-runtime.txt"
            ),
            str(
                REMOTE_BASE_DIRECTORY
                / "bin"
                / "setup_ids_inference_runtime.sh"
            ),
            str(
                REMOTE_BASE_DIRECTORY
                / "aws_ids_testbed_10"
                / "ids_input_agent.py"
            ),
            str(
                REMOTE_BASE_DIRECTORY
                / "aws_ids_testbed_10"
                / "ids_inference_agent.py"
            ),
            str(
                REMOTE_BASE_DIRECTORY
                / "ids_detector"
                / "lite_v6"
                / "evaluation"
                / "evaluator.py"
            ),
            str(
                REMOTE_BASE_DIRECTORY
                / "Lite_V9_CICIoT2023"
                / "outputs"
                / "models"
                / "best_model.pt"
            ),
            str(
                REMOTE_BASE_DIRECTORY
                / "Lite_V9_CICIoT2023"
                / "outputs"
                / "data"
                / "standard_scaler.pkl"
            ),
        }

        self.assertTrue(expected_paths.issubset(remote_paths))

        for deployment_file in manifest:
            self.assertTrue(deployment_file.source_path.is_file())
            self.assertGreater(
                deployment_file.source_path.stat().st_size,
                0,
            )

        self.assertEqual(len(remote_paths), len(manifest))

        prohibited_parts = (
            "ids_receiver_app.py",
            "ids_pcap_to_csv",
            "X_train",
            "X_val",
            "X_test",
            ".venv",
            "env_aws_ids",
            ".pem",
        )

        for remote_path in remote_paths:
            self.assertFalse(
                any(
                    prohibited_part in remote_path
                    for prohibited_part in prohibited_parts
                ),
                remote_path,
            )

    def test_upload_uses_existing_ids_ssh_settings(self) -> None:
        with TemporaryDirectory() as directory:
            project_root = Path(directory)
            private_key = project_root / "test-key.pem"
            source_file = project_root / "runtime.py"
            private_key.write_text("key", encoding="utf-8")
            source_file.write_text("value = 1\n", encoding="utf-8")

            manifest = (
                RuntimeDeploymentFile(
                    source_path=source_file,
                    remote_path=(
                        REMOTE_BASE_DIRECTORY
                        / "aws_ids_testbed_10"
                        / "runtime.py"
                    ),
                ),
            )

            stdout = MagicMock()
            stdout.__iter__.return_value = iter(())
            stdout_channel = MagicMock()
            stdout_channel.recv_exit_status.return_value = 0
            stdout.channel = stdout_channel
            stderr = MagicMock()

            sftp = MagicMock()
            sftp.__enter__.return_value = sftp
            sftp.__exit__.return_value = False

            ssh_client = MagicMock()
            ssh_client.exec_command.return_value = (
                MagicMock(),
                stdout,
                stderr,
            )
            ssh_client.open_sftp.return_value = sftp

            with (
                patch(
                    "aws_ids_testbed_10.ids_inference_deployment."
                    "build_runtime_deployment_manifest",
                    return_value=manifest,
                ),
                patch(
                    "aws_ids_testbed_10.ids_inference_deployment."
                    "get_ssh_username",
                    return_value="ubuntu",
                ),
                patch(
                    "aws_ids_testbed_10.ids_inference_deployment."
                    "get_private_key_path",
                    return_value=private_key,
                ),
                patch(
                    "aws_ids_testbed_10.ids_inference_deployment."
                    "get_public_host",
                    return_value="203.0.113.10",
                ),
                patch(
                    "paramiko.SSHClient",
                    return_value=ssh_client,
                ),
                patch("paramiko.AutoAddPolicy"),
            ):
                status = upload_ids_inference_runtime(
                    project_root
                )

            self.assertEqual(status, 0)
            ssh_client.connect.assert_called_once_with(
                hostname="203.0.113.10",
                username="ubuntu",
                key_filename=str(private_key),
                timeout=30,
            )
            sftp.put.assert_called_once_with(
                str(source_file),
                str(manifest[0].remote_path),
            )
            sftp.chmod.assert_called_once_with(
                str(
                    REMOTE_BASE_DIRECTORY
                    / "bin"
                    / "setup_ids_inference_runtime.sh"
                ),
                0o700,
            )
            ssh_client.close.assert_called_once()

    def test_remote_setup_validates_agents_and_model(self) -> None:
        project_root = Path("/project")
        runner = MagicMock()
        runner.run_command.return_value = 0

        with (
            patch(
                "aws_ids_testbed_10.ids_inference_deployment."
                "get_ssh_username",
                return_value="ubuntu",
            ),
            patch(
                "aws_ids_testbed_10.ids_inference_deployment."
                "get_private_key_path",
                return_value=Path("/project/key.pem"),
            ),
            patch(
                "aws_ids_testbed_10.ids_inference_deployment."
                "get_public_host",
                return_value="203.0.113.10",
            ),
            patch(
                "aws_ids_testbed_10.ids_inference_deployment."
                "RemoteRunner",
                return_value=runner,
            ) as runner_class,
        ):
            status = setup_uploaded_ids_inference_runtime(
                project_root
            )

        self.assertEqual(status, 0)
        runner_class.assert_called_once_with(
            username="ubuntu",
            private_key_path=Path("/project/key.pem"),
        )
        runner.run_command.assert_called_once()

        call = runner.run_command.call_args.kwargs
        self.assertEqual(call["host"], "203.0.113.10")
        self.assertIn(
            "setup_ids_inference_runtime.sh",
            call["command"],
        )
        self.assertIn("ids_input_agent.py", call["command"])
        self.assertIn("ids_inference_agent.py", call["command"])
        self.assertIn(
            "_load_lite_v6_trained_model('cpu')",
            call["command"],
        )

    def test_combined_deployment_stops_after_upload_failure(self) -> None:
        project_root = Path("/project")

        with (
            patch(
                "aws_ids_testbed_10.ids_inference_deployment."
                "upload_ids_inference_runtime",
                return_value=1,
            ),
            patch(
                "aws_ids_testbed_10.ids_inference_deployment."
                "setup_uploaded_ids_inference_runtime"
            ) as setup,
        ):
            status = deploy_ids_inference_runtime(project_root)

        self.assertEqual(status, 1)
        setup.assert_not_called()

    def test_combined_deployment_runs_setup_after_upload(self) -> None:
        project_root = Path("/project")

        with (
            patch(
                "aws_ids_testbed_10.ids_inference_deployment."
                "upload_ids_inference_runtime",
                return_value=0,
            ),
            patch(
                "aws_ids_testbed_10.ids_inference_deployment."
                "setup_uploaded_ids_inference_runtime",
                return_value=0,
            ) as setup,
        ):
            status = deploy_ids_inference_runtime(project_root)

        self.assertEqual(status, 0)
        setup.assert_called_once_with(project_root)


if __name__ == "__main__":
    unittest.main()
