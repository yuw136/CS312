"""Offline-history and run-identity checks; no Torch, W&B or GPU required."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from colab.a1_runner import JsonlLogger, guard_experiment, read_history


class HistoryTests(unittest.TestCase):
    def setUp(self):
        workspace = Path(__file__).resolve().parents[1]
        self.temporary = tempfile.TemporaryDirectory(prefix="test-colab-", dir=workspace)
        self.root = Path(self.temporary.name)
        self.assertEqual(self.root.resolve().parent, workspace)
        self.addCleanup(self.temporary.cleanup)
        self.path = self.root / "history.jsonl"

    def write_rows(self, rows):
        self.path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_logger_preserves_real_log_train_eval_and_rms(self):
        real_log = Mock(return_value="logged")
        wandb = SimpleNamespace(log=real_log)
        logger = JsonlLogger(self.path, wandb)
        logger.setup(SimpleNamespace(step=0))
        training = {"optimizer_step": 24, "train_loss": 3.5,
                    "logging/rms/global/gradient": 0.02}
        self.assertEqual(wandb.log(training), "logged")
        wandb.log({"optimizer_step": 24, "val_loss": 3.7})
        self.assertEqual(real_log.call_count, 2)
        self.assertEqual(read_history(self.path), [{**training, "val_loss": 3.7}])
        logger.close()
        self.assertIs(wandb.log, real_log)
        logger.close()  # Safe outer-finally cleanup after trainer cleanup.

    def test_resume_removes_every_uncheckpointed_row(self):
        self.write_rows([{"optimizer_step": 2, "train_loss": 4.0},
                         {"optimizer_step": 3, "val_loss": 9.0},
                         {"optimizer_step": 4, "train_loss": 8.0}])
        logger = JsonlLogger(self.path, SimpleNamespace(log=Mock()))
        logger.setup(SimpleNamespace(step=3))
        logger.close()
        self.assertEqual(read_history(self.path), [{"optimizer_step": 2, "train_loss": 4.0}])
        logger.setup(SimpleNamespace(step=0))
        logger.close()
        self.assertEqual(read_history(self.path), [])

    def test_merges_duplicate_steps_and_honors_checkpoint_ceiling(self):
        self.write_rows([{"optimizer_step": 1, "train_loss": 8.0},
                         {"optimizer_step": 0, "val_loss": 7.0},
                         {"optimizer_step": 1, "train_loss": 6.0},
                         {"optimizer_step": 1, "val_loss": 5.0}])
        self.assertEqual(read_history(self.path), [
            {"optimizer_step": 0, "val_loss": 7.0},
            {"optimizer_step": 1, "train_loss": 6.0, "val_loss": 5.0}])
        self.assertEqual(read_history(self.path, checkpoint_ceiling=1), [
            {"optimizer_step": 0, "val_loss": 7.0}])

    def test_tolerates_torn_final_line_but_rejects_corrupt_interior(self):
        self.path.write_text('{"optimizer_step": 0, "val_loss": 5}\n{"optim', encoding="utf-8")
        self.assertEqual(read_history(self.path), [{"optimizer_step": 0, "val_loss": 5}])
        self.path.write_text('broken\n{"optimizer_step": 0}\n', encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError):
            read_history(self.path)

    def test_identity_guard_allows_resume_and_rejects_changed_run(self):
        guard_experiment(self.root, {"lr": 0.003, "tags": ("a1",)}, "abc")
        guard_experiment(self.root, {"lr": 0.003, "tags": ["a1"]}, "abc")
        with self.assertRaisesRegex(ValueError, "new run_name"):
            guard_experiment(self.root, {"lr": 0.03, "tags": ["a1"]}, "abc")
        with self.assertRaisesRegex(ValueError, "Git SHA"):
            guard_experiment(self.root, {"lr": 0.003, "tags": ["a1"]}, "def")


if __name__ == "__main__":
    unittest.main()
