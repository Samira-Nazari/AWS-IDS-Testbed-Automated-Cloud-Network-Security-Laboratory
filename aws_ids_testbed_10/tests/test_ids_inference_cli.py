import argparse
import unittest
from unittest.mock import patch

from aws_ids_testbed_10 import cli


class TestIdsInferenceCli(unittest.TestCase):
    def test_deploy_inference_runtime_command_is_registered(self) -> None:
        parser = cli.build_parser()
        arguments = parser.parse_args(
            ["deploy-ids-inference-runtime"]
        )

        self.assertIs(
            arguments.handler,
            cli.deploy_ids_inference_runtime_command,
        )

    def test_deploy_command_uses_project_root(self) -> None:
        with patch(
            "aws_ids_testbed_10.cli.deploy_ids_inference_runtime",
            return_value=0,
        ) as deploy:
            status = cli.deploy_ids_inference_runtime_command(
                argparse.Namespace()
            )

        self.assertEqual(status, 0)
        deploy.assert_called_once_with(cli.PROJECT_ROOT)


if __name__ == "__main__":
    unittest.main()
