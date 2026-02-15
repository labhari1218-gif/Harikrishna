"""Tests for Component 2 I/O helper utilities."""

from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from pathlib import Path

from component2.io_utils import ensure_positive_max_claims, make_run_dir


class IoUtilsTests(unittest.TestCase):
    def test_ensure_positive_max_claims_requires_positive_value(self) -> None:
        with self.assertRaises(ValueError):
            ensure_positive_max_claims(0)

    def test_make_run_dir_handles_same_timestamp_collision(self) -> None:
        fixed_now = dt.datetime(2026, 2, 10, 8, 30, 0, 123456, tzinfo=dt.timezone.utc)
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            first = make_run_dir(output_root=output_root, milestone="m1", now=fixed_now)
            second = make_run_dir(output_root=output_root, milestone="m1", now=fixed_now)

            self.assertNotEqual(first, second)
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())
            self.assertTrue(second.name.endswith("_01"))


if __name__ == "__main__":
    unittest.main()
