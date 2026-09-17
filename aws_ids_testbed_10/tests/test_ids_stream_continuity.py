import unittest

from aws_ids_testbed_10.ids_stream_continuity import (
    plan_continuous_windows,
)


class TestStreamContinuity(unittest.TestCase):
    def test_window_and_buffer_calculations(self) -> None:
        cases = [
            # rows, windows, next offset, retained
            (20, 0, 0, 20),
            (29, 0, 0, 29),
            (30, 1, 5, 25),
            (31, 1, 5, 26),
            (35, 2, 10, 25),
            (45, 4, 20, 25),
            (50, 5, 25, 25),
        ]

        for total_rows, windows, next_offset, retained in cases:
            with self.subTest(total_rows=total_rows):
                plan = plan_continuous_windows(total_rows)

                self.assertEqual(plan.window_count, windows)
                self.assertEqual(plan.next_start_offset, next_offset)
                self.assertEqual(plan.retained_rows, retained)
                self.assertEqual(plan.can_run_inference, windows > 0)

    def test_negative_row_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            plan_continuous_windows(-1)


if __name__ == "__main__":
    unittest.main()
