"""PyTorch model calls run in a worker process that exits, so MPS's graph cache cannot build up.

No model is loaded; the worker runs small functions from tests/isolation_helpers.py.
"""
import os
import unittest
from unittest.mock import patch

from cloudir.ai_models.model_worker import ISOLATE_ENV
from tests import isolation_helpers as helpers


class ModelWorkerTests(unittest.TestCase):
    def test_calls_stay_in_process_unless_isolation_is_on(self):
        with patch.dict(os.environ, {ISOLATE_ENV: "0"}):
            self.assertEqual(helpers.process_id("a", extra={})["pid"], os.getpid())

    def test_isolated_calls_run_in_a_worker_and_return_the_result(self):
        with patch.dict(os.environ, {ISOLATE_ENV: "1"}):
            result = helpers.process_id("turn 1", extra={"rows": [1, 2]})

        self.assertNotEqual(result["pid"], os.getpid())
        self.assertEqual((result["tag"], result["extra"]), ("turn 1", {"rows": [1, 2]}))

    def test_worker_errors_keep_their_type_so_retries_still_work(self):
        with patch.dict(os.environ, {ISOLATE_ENV: "1"}), \
             self.assertRaisesRegex(helpers.ExtractionLikeError, "rows unreadable"):
            helpers.fail("rows unreadable")

    def test_a_crashed_worker_is_reported(self):
        with patch.dict(os.environ, {ISOLATE_ENV: "1"}), \
             self.assertRaisesRegex(RuntimeError, "exited with code 3"):
            helpers.crash(3)


if __name__ == "__main__":
    unittest.main()
