"""Static contract tests for Component 3 autopilot shell scripts."""

from __future__ import annotations

from pathlib import Path
import unittest


class AutopilotScriptsContractTests(unittest.TestCase):
    def test_autopilot_runner_requires_gate_file_for_pass(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        script = (repo_root / "scripts" / "autopilot_component3.sh").read_text(encoding="utf-8")
        self.assertIn('if stage_result="$(run_stage "${stage}" "${iter}")"; then', script)
        self.assertNotIn("gate_ok=1", script)

    def test_stage_scripts_export_src_pythonpath(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        rel_paths = [
            "scripts/overfit_200.sh",
            "scripts/claim_only_baseline.sh",
            "scripts/phase4_lora_or_unfreeze.sh",
            "scripts/medium_ab_eval.sh",
        ]
        for rel in rel_paths:
            content = (repo_root / rel).read_text(encoding="utf-8")
            self.assertIn('export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"', content, msg=rel)

    def test_phase4_installs_peft_with_interpreter_pip(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        script = (repo_root / "scripts" / "phase4_lora_or_unfreeze.sh").read_text(encoding="utf-8")
        self.assertIn("run_python python -m pip install peft", script)


if __name__ == "__main__":
    unittest.main()
