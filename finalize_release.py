"""One-shot final release finalizer for the IITM RL project.

Run from the repository root after the final A2C Rep-C model exists at the
expected local training path and the corrected final report has been copied to
Project_Report_DA25M579_V030.docx. The script is deliberately strict and stops
before destructive cleanup if required inputs or validation checks fail.

On success it removes itself so the helper is not part of the final release.
"""
from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAINING = ROOT / "training_pipelines"
SUBMISSIONS = ROOT / "submissions"
RESULTS = ROOT / "results"

A2C_SOURCE_MODEL = TRAINING / "models" / "a2c_rep_c_experiments" / "screen_lr5e-04_g0.99_lam0.95_ent0.001_n64_vf0.5_shape_seed20260902" / "final_model.zip"
A2C_FINAL_MODEL = SUBMISSIONS / "a2c" / "model.zip"
REPORT = ROOT / "Project_Report_DA25M579_V030.docx"
NOTEBOOK = ROOT / "RL_Project_Final.ipynb"
README = ROOT / "README.md"

FINAL_SUBMISSIONS = {
    "ppo": ("policy.py", "model.zip", "metadata.json"),
    "dqn": ("policy.py", "model.zip", "metadata.json"),
    "neural_sarsa": ("policy.py", "policy_state.pt", "metadata.json"),
    "a2c": ("policy.py", "model.zip", "metadata.json"),
    "double_dqn": ("policy.py", "model.zip", "metadata.json"),
}

KEEP_TRAINING_ROOT = {"assigned_config.json", "src", "training_scripts", "training_utils"}
KEEP_TRAINING_SCRIPTS = {
    "common.py",
    "train_ppo_rep_b_experiment.py",
    "train_dqn.py",
    "train_neural_sarsa_final.py",
    "train_a2c_rep_c.py",
    "train_double_dqn_experiment.py",
}
KEEP_SRC = {
    "algorithms/neural_sarsa.py",
    "algorithms/common/checkpoint.py",
    "algorithms/common/env_factory.py",
    "algorithms/common/networks.py",
    "algorithms/common/replay_buffer.py",
    "algorithms/common/schedules.py",
    "environment/action_codec.py",
    "environment/wrappers.py",
    "features/normalizer.py",
    "features/observation.py",
    "features/engineered.py",
    "features/representation_c.py",
}
KEEP_RESULTS = {"final_results.csv", "ppo_learning_curve.png", "dqn_learning_curve.png", "a2c_learning_curve.png"}


def fail(message: str) -> None:
    raise RuntimeError(message)


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    if result.returncode != 0:
        fail(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def check_assigned_config() -> None:
    payload = json.loads((TRAINING / "assigned_config.json").read_text(encoding="utf-8"))
    if payload.get("variant_id") != "V030":
        fail(f"Unexpected variant_id: {payload.get('variant_id')}")
    if payload.get("config_fingerprint") != "5566e180ff842b10":
        fail(f"Unexpected config fingerprint: {payload.get('config_fingerprint')}")
    print("OK assigned configuration")


def check_inputs() -> None:
    for path, label in [(A2C_SOURCE_MODEL, "A2C source model"), (REPORT, "final report"), (NOTEBOOK, "final notebook")]:
        if not path.exists():
            fail(f"Missing {label}: {path}")
    if A2C_SOURCE_MODEL.stat().st_size < 100_000:
        fail("A2C source model is unexpectedly small")
    print("OK required finalization inputs")


def replace_a2c_model() -> None:
    A2C_FINAL_MODEL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(A2C_SOURCE_MODEL, A2C_FINAL_MODEL)
    if A2C_FINAL_MODEL.stat().st_size != A2C_SOURCE_MODEL.stat().st_size:
        fail("A2C model copy size mismatch")
    print("OK A2C final model copied")


def validate_text_files() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if nb.get("nbformat") != 4:
        fail("Final notebook is not nbformat 4")
    nb_text = "\n".join("".join(c.get("source", [])) for c in nb.get("cells", []))
    # These checks intentionally verify the course-required content semantically
    # against the final notebook's actual headings/phrasing.
    required_any = [
        ("Assigned parameter variant",),
        ("Reproducible final training pipelines",),
        ("Final public leaderboard comparison",),
        ("Final policy and model mapping", "Final policy/artifact mapping"),
        ("Course-supplied policy validation",),
        ("A2C + Representation C",),
        ("94660.12",),
        ("99566.57",),
    ]
    for alternatives in required_any:
        if not any(token in nb_text for token in alternatives):
            fail(f"Final notebook missing required text: {' or '.join(alternatives)}")
    for token in ["111,216.54", "111216.54", "114047.62", "Double-DQN pipeline use 76-dimensional", "public score pending"]:
        if token in nb_text:
            fail(f"Stale notebook text detected: {token}")

    readme = README.read_text(encoding="utf-8")
    for token in ["A2C + Representation C", "94,660.12", "99,566.57", "V030", "RL_Project_Final.ipynb"]:
        if token not in readme:
            fail(f"README missing final-release text: {token}")
    for token in ["111,216.54", "dueling joint-action Double DQN model", "A2C — public score 114,047.62"]:
        if token in readme:
            fail(f"Stale README text detected: {token}")
    print("OK notebook and README checks")


def validate_submission_layout() -> None:
    actual = {p.name for p in SUBMISSIONS.iterdir() if p.is_dir()}
    expected = set(FINAL_SUBMISSIONS)
    if actual != expected:
        fail(f"Final submission directories mismatch: expected {expected}, found {actual}")
    for name, files in FINAL_SUBMISSIONS.items():
        for filename in files[:2]:
            path = SUBMISSIONS / name / filename
            if not path.exists() or path.stat().st_size == 0:
                fail(f"Missing/empty artifact: {path}")
        metadata = SUBMISSIONS / name / files[2]
        if not metadata.exists():
            fail(f"Missing metadata: {metadata}")
    print("OK submission layout")


def cleanup() -> None:
    stale = SUBMISSIONS / "a2c_rep_c_candidate_v030"
    if stale.exists():
        shutil.rmtree(stale)
    for child in list(TRAINING.iterdir()):
        if child.name not in KEEP_TRAINING_ROOT:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    scripts_dir = TRAINING / "training_scripts"
    for child in list(scripts_dir.iterdir()):
        if child.name not in KEEP_TRAINING_SCRIPTS and child.name != "__init__.py":
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    for path in [TRAINING / "models", TRAINING / "logs", TRAINING / "eval_results", ROOT / "tuning_results"]:
        if path.exists():
            shutil.rmtree(path)
    src = TRAINING / "src"
    for path in sorted(src.rglob("*"), reverse=True):
        if path.is_dir() or path.name == "__init__.py":
            continue
        if path.relative_to(src).as_posix() not in KEEP_SRC:
            path.unlink()
    for path in sorted(src.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    if RESULTS.exists():
        for child in list(RESULTS.iterdir()):
            if child.name not in KEEP_RESULTS:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
    print("OK repository cleanup")


def validate_policy_sources() -> None:
    for name in FINAL_SUBMISSIONS:
        path = SUBMISSIONS / name / "policy.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"step", "reset"} and isinstance(node.func.value, ast.Name) and node.func.value.id == "env":
                    fail(f"Forbidden env.{node.func.attr}() in {path} line {node.lineno}")
        run([sys.executable, "policy_validation_tests.py", str(path), "--max-seconds-per-call", "0.25"])
    print("OK all five course policy validators")


def full_holdout_validation() -> None:
    helper = ROOT / ".final_holdout_validation_tmp.py"
    helper.write_text(
        """import importlib.util\nfrom pathlib import Path\nfrom evaluation import evaluate_policy, summarise_overall\nROOT=Path(__file__).resolve().parent\nPOLICIES={\n'PPO':ROOT/'submissions/ppo/policy.py',\n'DQN':ROOT/'submissions/dqn/policy.py',\n'Neural SARSA':ROOT/'submissions/neural_sarsa/policy.py',\n'A2C':ROOT/'submissions/a2c/policy.py',\n'Double DQN':ROOT/'submissions/double_dqn/policy.py',\n}\nfor name,path in POLICIES.items():\n spec=importlib.util.spec_from_file_location('final_'+name.replace(' ','_'),path)\n mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n per,summary=evaluate_policy(mod.run_policy, progress=False)\n overall=summarise_overall(per)\n print(f\"{name}: mean={overall['mean_cost']:.2f} std={overall['std_cost']:.2f} service={overall['mean_service_level']:.6f} episodes={overall['n_episodes']}\")\n if overall['n_episodes'] != 200: raise SystemExit(f'{name}: wrong episode count')\n""",
        encoding="utf-8",
    )
    try:
        run([sys.executable, str(helper)])
    finally:
        helper.unlink(missing_ok=True)
    print("OK full 200-episode official-style holdout for all five policies")


def main() -> int:
    print("=== IITM RL final release ===", flush=True)
    check_assigned_config()
    check_inputs()
    replace_a2c_model()
    validate_text_files()
    validate_submission_layout()
    cleanup()
    validate_submission_layout()
    validate_policy_sources()
    full_holdout_validation()
    print("\nFINAL RELEASE VALIDATION PASSED", flush=True)
    Path(__file__).unlink(missing_ok=True)
    print("Temporary finalizer removed itself.", flush=True)
    print("Inspect git status/diff, then commit and push the branch.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
