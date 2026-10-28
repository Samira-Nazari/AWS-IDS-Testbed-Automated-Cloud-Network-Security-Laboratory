import argparse
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import yaml
from botocore.exceptions import ClientError

from aws_ids_testbed_11 import benign_generator_lifecycle, cli, ec2_lab
from aws_ids_testbed_11.inventory import load_inventory, save_inventory


INSTANCE_ID = "i-0123456789abcdef0"


def aws_instance(instance_id=INSTANCE_ID):
    return {
        "InstanceId": instance_id,
        "State": {"Name": "running"},
        "PublicIpAddress": "203.0.113.10",
        "PrivateIpAddress": "10.0.1.10",
        "PublicDnsName": "ec2.example.test",
    }


class TestRefreshInstanceInfo(unittest.TestCase):
    def test_success_returns_existing_fields_and_requests_saved_id(self):
        ec2 = MagicMock()
        ec2.describe_instances.return_value = {
            "Reservations": [{"Instances": [aws_instance()]}]
        }
        project_root = Path("/unused-project")

        with patch.object(ec2_lab, "create_ec2_client", return_value=ec2):
            result = ec2_lab.refresh_instance_info(project_root, INSTANCE_ID)

        self.assertEqual(
            result,
            {
                "instance_id": INSTANCE_ID,
                "state": "running",
                "public_ip": "203.0.113.10",
                "private_ip": "10.0.1.10",
                "public_dns": "ec2.example.test",
            },
        )
        ec2.describe_instances.assert_called_once_with(InstanceIds=[INSTANCE_ID])

    def test_missing_optional_addresses_return_none(self):
        ec2 = MagicMock()
        ec2.describe_instances.return_value = {
            "Reservations": [{"Instances": [{
                "InstanceId": INSTANCE_ID,
                "State": {"Name": "terminated"},
            }]}]
        }

        with patch.object(ec2_lab, "create_ec2_client", return_value=ec2):
            result = ec2_lab.refresh_instance_info(Path("/unused"), INSTANCE_ID)

        self.assertEqual(result["state"], "terminated")
        self.assertIsNone(result["public_ip"])
        self.assertIsNone(result["private_ip"])
        self.assertIsNone(result["public_dns"])

    def test_absent_requested_instance_raises_lookup_error(self):
        responses = (
            {},
            {"Reservations": None},
            {"Reservations": []},
            {"Reservations": [{}]},
            {"Reservations": [{"Instances": None}]},
            {"Reservations": [{"Instances": []}]},
            {"Reservations": [{"Instances": [aws_instance("i-other")]}]},
        )
        for response in responses:
            with self.subTest(response=response):
                ec2 = MagicMock()
                ec2.describe_instances.return_value = response
                with patch.object(ec2_lab, "create_ec2_client", return_value=ec2):
                    with self.assertRaises(ec2_lab.InstanceNotFoundError) as caught:
                        ec2_lab.refresh_instance_info(Path("/unused"), INSTANCE_ID)

                self.assertIsInstance(caught.exception, LookupError)
                self.assertEqual(caught.exception.instance_id, INSTANCE_ID)
                self.assertIn(INSTANCE_ID, str(caught.exception))

    def test_searches_all_reservations_for_requested_id(self):
        ec2 = MagicMock()
        ec2.describe_instances.return_value = {
            "Reservations": [
                {},
                {"Instances": [aws_instance("i-other")]},
                {"Instances": [aws_instance()]},
            ]
        }

        with patch.object(ec2_lab, "create_ec2_client", return_value=ec2):
            result = ec2_lab.refresh_instance_info(Path("/unused"), INSTANCE_ID)

        self.assertEqual(result["instance_id"], INSTANCE_ID)

    def test_aws_not_found_error_becomes_instance_not_found(self):
        ec2 = MagicMock()
        error = ClientError(
            {"Error": {"Code": "InvalidInstanceID.NotFound", "Message": "Absent"}},
            "DescribeInstances",
        )
        ec2.describe_instances.side_effect = error

        with patch.object(ec2_lab, "create_ec2_client", return_value=ec2):
            with self.assertRaises(ec2_lab.InstanceNotFoundError) as caught:
                ec2_lab.refresh_instance_info(Path("/unused"), INSTANCE_ID)

        self.assertEqual(caught.exception.instance_id, INSTANCE_ID)

    def test_other_aws_errors_propagate_unchanged(self):
        for code in ("UnauthorizedOperation", "AuthFailure", "InvalidInstanceID.Malformed"):
            with self.subTest(code=code):
                error = ClientError(
                    {"Error": {"Code": code, "Message": "Failure"}},
                    "DescribeInstances",
                )
                ec2 = MagicMock()
                ec2.describe_instances.side_effect = error

                with patch.object(ec2_lab, "create_ec2_client", return_value=ec2):
                    with self.assertRaises(ClientError) as caught:
                        ec2_lab.refresh_instance_info(Path("/unused"), INSTANCE_ID)

                self.assertIs(caught.exception, error)


class TestRefreshCommands(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.project_root = Path(self.temporary_directory.name)
        (self.project_root / "config.yaml").write_text(
            "aws_region: us-east-1\n", encoding="utf-8"
        )

    def write_inventory(self, instances):
        inventory = {"instances": instances, "notes": "keep unrelated metadata"}
        save_inventory(self.project_root, inventory)
        return inventory

    def test_missing_instances_leave_inventory_unchanged_for_all_three_commands(self):
        commands = (
            ("victim", cli.refresh_victim),
            ("attacker", cli.refresh_attacker),
            ("ids", cli.refresh_ids),
        )
        for role, command in commands:
            with self.subTest(role=role):
                self.write_inventory({role: {
                    "role": role,
                    "name": role + "-ec2",
                    "instance_id": INSTANCE_ID,
                    "state": "terminated",
                    "previous_state": "running",
                    "public_ip": None,
                }})
                before = (self.project_root / "inventory.yaml").read_bytes()
                output = io.StringIO()
                ec2 = MagicMock()
                ec2.describe_instances.return_value = {"Reservations": []}

                with (
                    patch.object(cli, "PROJECT_ROOT", self.project_root),
                    patch.object(ec2_lab, "create_ec2_client", return_value=ec2),
                    patch.object(cli, "update_instance") as update,
                    redirect_stdout(output),
                ):
                    status = command(argparse.Namespace())

                self.assertEqual(status, 1)
                update.assert_not_called()
                self.assertEqual(
                    (self.project_root / "inventory.yaml").read_bytes(), before
                )
                self.assertIn("Unable to refresh " + role + ":", output.getvalue())
                self.assertIn(INSTANCE_ID, output.getvalue())
                self.assertIn("region", output.getvalue().lower())
                self.assertIn("account", output.getvalue().lower())
                self.assertIn("Saved inventory entry was preserved.", output.getvalue())

    def test_successful_role_refresh_preserves_metadata_and_prints_yaml(self):
        self.write_inventory({
            "victim": {
                "role": "victim",
                "name": "victim-ec2",
                "instance_id": INSTANCE_ID,
                "state": "pending",
                "previous_state": "stopped",
                "custom": "keep",
            },
            "attacker": {"instance_id": "i-untouched", "state": "stopped"},
        })
        ec2 = MagicMock()
        ec2.describe_instances.return_value = {
            "Reservations": [{"Instances": [aws_instance()]}]
        }
        output = io.StringIO()

        with (
            patch.object(cli, "PROJECT_ROOT", self.project_root),
            patch.object(ec2_lab, "create_ec2_client", return_value=ec2),
            redirect_stdout(output),
        ):
            status = cli.refresh_role("victim")

        self.assertEqual(status, 0)
        inventory = load_inventory(self.project_root)
        victim = inventory["instances"]["victim"]
        self.assertEqual(victim["role"], "victim")
        self.assertEqual(victim["name"], "victim-ec2")
        self.assertEqual(victim["state"], "running")
        self.assertEqual(victim["previous_state"], "stopped")
        self.assertEqual(victim["custom"], "keep")
        self.assertEqual(victim["public_ip"], "203.0.113.10")
        self.assertEqual(inventory["instances"]["attacker"]["state"], "stopped")
        self.assertEqual(inventory["notes"], "keep unrelated metadata")
        self.assertEqual(yaml.safe_load(output.getvalue()), victim)

    def test_generator_refresh_continues_after_missing_instance_and_reports_failure(self):
        roles = benign_generator_lifecycle.BENIGN_GENERATOR_ROLES
        original = self.write_inventory({
            role: {
                "role": role,
                "name": role + "-ec2",
                "instance_id": "i-generator-" + str(index),
                "state": "pending",
                "public_ip": "old-address",
            }
            for index, role in enumerate(roles)
        })
        missing_id = original["instances"][roles[1]]["instance_id"]
        ec2 = MagicMock()

        def describe_instances(*, InstanceIds):
            instance_id = InstanceIds[0]
            if instance_id == missing_id:
                return {"Reservations": []}
            return {"Reservations": [{"Instances": [aws_instance(instance_id)]}]}

        ec2.describe_instances.side_effect = describe_instances
        output = io.StringIO()

        with (
            patch.object(ec2_lab, "create_ec2_client", return_value=ec2),
            redirect_stdout(output),
        ):
            status = benign_generator_lifecycle.refresh_benign_generators(self.project_root)

        self.assertEqual(status, 1)
        self.assertEqual(ec2.describe_instances.call_count, len(roles))
        inventory = load_inventory(self.project_root)
        self.assertEqual(inventory["instances"][roles[1]], original["instances"][roles[1]])
        for role in (roles[0], *roles[2:]):
            self.assertEqual(inventory["instances"][role]["state"], "running")
            self.assertEqual(inventory["instances"][role]["name"], role + "-ec2")
            self.assertEqual(inventory["instances"][role]["public_ip"], "203.0.113.10")
        self.assertEqual(inventory["notes"], "keep unrelated metadata")
        self.assertIn(missing_id, output.getvalue())
        self.assertIn("Inventory refresh incomplete", output.getvalue())
        self.assertNotIn("[benign-generators] Inventory refreshed.", output.getvalue())
        ec2.run_instances.assert_not_called()
        ec2.terminate_instances.assert_not_called()

    def test_all_successful_generator_refreshes_report_success(self):
        roles = benign_generator_lifecycle.BENIGN_GENERATOR_ROLES
        self.write_inventory({
            role: {"instance_id": "i-generator-" + str(index), "name": role}
            for index, role in enumerate(roles)
        })
        ec2 = MagicMock()
        ec2.describe_instances.side_effect = lambda *, InstanceIds: {
            "Reservations": [{"Instances": [aws_instance(InstanceIds[0])]}]
        }
        output = io.StringIO()

        with (
            patch.object(ec2_lab, "create_ec2_client", return_value=ec2),
            redirect_stdout(output),
        ):
            status = benign_generator_lifecycle.refresh_benign_generators(self.project_root)

        self.assertEqual(status, 0)
        self.assertIn("[benign-generators] Inventory refreshed.", output.getvalue())
        self.assertNotIn("Inventory refresh incomplete", output.getvalue())
        for role in roles:
            self.assertEqual(load_inventory(self.project_root)["instances"][role]["state"], "running")


if __name__ == "__main__":
    unittest.main()
