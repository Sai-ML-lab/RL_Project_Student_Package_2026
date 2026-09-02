"""Finalize, validate, and clean the IITM RL project release.

Run from the repository root after placing the final A2C Rep-C model at the
path printed below and replacing the stale report with the provided final
report. The script is intentionally strict: it stops before cleanup if required
inputs are missing or if any policy validator fails.

The script does not change the official environment, cost function, evaluator,
or assigned configuration.
"""
from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAINING = ROOT / "training_pipelines"
SUBMISSIONS = ROOT / "submissions"
RESULTS = ROOT / "results"

A2C_SOURCE_MODEL = (
    TRAINING
    / "models"
    / "a2c_rep_c_experiments"
    / "screen_lr5e-04_g0.99_lam0.95_ent0.001_n64_vf0.5_shape_seed20260902"
    / "final_model.zip"
)
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

# Final release source files. __init__.py files are retained to preserve
# package semantics; all other source files must be explicit.
KEEP_TRAINING_SCRIPTS = {
    "common.py",
    "train_ppo_rep_b_experiment.py",
    "train_dqn.py",
    "train_neural_sarsa_final.py",
    "train_a2c_rep_c.py",
    "train_double_dqn_experiment.py",
}
KEEP_TRAINING_ROOT = {"assigned_config.json", "src", "training_scripts", "training_utils"}
KEEP_SRC = {
    "algorithms/neural_sarsa.py",
    "algorithms/common/checkpoint.py",
    "algorithms/common/networks.py",
    "algorithms/common/replay_buffer.py",
    "algorithms/common/schedules.py",
    "environment/action_codec.py",
    "environment/wrappers.py",
    "features/observation.py",
    "features/engineered.py",
    "features/representation_c.py",
}
KEEP_RESULTS = {
    "final_results.csv",
    "ppo_learning_curve.png",
    "dqn_learning_curve.png",
    "a2c_learning_curve.png",
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=ROOT, check=False)
    if completed.returncode != 0:
        fail(f"Command failed with exit code {completed.returncode}: {' '.join(cmd)}")


def check_assigned_config() -> None:
    payload = json.loads((TRAINING / "assigned_config.json").read_text(encoding="utf-8"))
    if payload.get("variant_id") != "V030":
        fail(f"Unexpected variant_id: {payload.get('variant_id')}")
    if payload.get("config_fingerprint") != "5566e180ff842b10":
        fail(f"Unexpected config fingerprint: {payload.get('config_fingerprint')}")
    print("OK assigned configuration: V030 / 5566e180ff842b10")


def check_inputs() -> None:
    for path, label in [(A2C_SOURCE_MODEL, "A2C Rep-C source model"), (REPORT, "final report"), (NOTEBOOK, "final notebook")]:
        if not path.exists():
            fail(f"Missing {label}: {path}")
    if A2C_SOURCE_MODEL.stat().st_size < 100_000:
        fail(f"A2C model is unexpectedly small: {A2C_SOURCE_MODEL.stat().st_size} bytes")
    print(f"OK input model: {A2C_SOURCE_MODEL}")
    print(f"OK final report: {REPORT}")
    print(f"OK final notebook: {NOTEBOOK}")


def replace_a2c_model() -> None:
    A2C_FINAL_MODEL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(A2C_SOURCE_MODEL, A2C_FINAL_MODEL)
    if A2C_FINAL_MODEL.stat().st_size != A2C_SOURCE_MODEL.stat().st_size:
        fail("A2C model copy size mismatch")
    print(f"OK A2C replacement model copied to {A2C_FINAL_MODEL}")


def validate_notebook() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if payload.get("nbformat") != 4:
        fail("Final notebook is not nbformat 4")
    text = "\n".join("".join(cell.get("source", [])) for cell in payload.get("cells", []))
    required = [
        "Assigned parameter variant",
        "Reproducible final training pipelines",
        "Final public leaderboard comparison",
        "Final policy/artifact mapping",
        "Course-supplied policy validation",
        "A2C + Representation C",
        "94660.12",
        "99566.57",
    ]
    for token in required:
        if token not in text:
            fail(f"Final notebook missing required content: {token}")
    forbidden = ["111,216.54", "111216.54", "114047.62", "Double-DQN pipeline use 76-dimensional", "public score pending"]
    for token in forbidden:
        if token in text:
            fail(f"Stale notebook content found: {token}")
    print("OK final notebook structure/content checks")


def validate_readme() -> None:
    text = README.read_text(encoding="utf-8")
    required = ["A2C + Representation C", "94,660.12", "99,566.57", "V030", "RL_Project_Final.ipynb"]
    for token in required:
        if token not in text:
            fail(f"README missing final-release content: {token}")
    forbidden = ["111,216.54", "dueling joint-action Double DQN model", "A2C — public score 114,047.62"]
    for token in forbidden:
        if token in text:
            fail(f"Stale README content found: {token}")
    print("OK README final-release checks")


def validate_submission_layout() -> None:
    actual_dirs = {p.name for p in SUBMISSIONS.iterdir() if p.is_dir()}
    expected_dirs = set(FINAL_SUBMISSIONS)
    if actual_dirs != expected_dirs:
        fail(f"Submission directories mismatch. Expected {expected_dirs}, found {actual_dirs}")
    for name, files in FINAL_SUBMISSIONS.items():
        folder = SUBMISSIONS / name
        for filename in files[:2]:
            path = folder / filename
            if not path.exists() or path.stat().st_size == 0:
                fail(f"Missing/empty final submission artifact: {path}")
        metadata = folder / files[2]
        if not metadata.exists():
            fail(f"Missing final metadata: {metadata}")
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        if name == "a2c":
            if payload.get("public_status") and payload.get("leaderboard_performance", {}).get("public_mean_cost") != 94660.12:
                fail("A2C metadata public score mismatch")
    print("OK final submission layout")


def cleanup() -> None:
    # Remove stale candidate submission directory after the final A2C copy.
    stale_submission = SUBMISSIONS / "a2c_rep_c_candidate_v030"
    if stale_submission.exists():
        shutil.rmtree(stale_submission)

    # Keep only final release source/launcher files.
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

    # Remove exploratory model/eval/tuning trees if they survived from the
    # tuning branch; they are not needed in the final deliverable.
    for path in [TRAINING / "models", TRAINING / "logs", TRAINING / "eval_results", ROOT / "tuning_results"]:
        if path.exists():
            shutil.rmtree(path)

    # Keep only core reusable source modules under src.
    src = TRAINING / "src"
    for path in sorted(src.rglob("*"), reverse=True):
        if path.is_dir():
            continue
        rel = path.relative_to(src).as_posix()
        if path.name == "__init__.py":
            continue
        if rel not in KEEP_SRC:
            path.unlink()

    # Remove empty directories left by source cleanup.
    for path in sorted(src.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()

    # Keep only final evidence files under results.
    if RESULTS.exists():
        for child in list(RESULTS.iterdir()):
            if child.name not in KEEP_RESULTS:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
    print("OK repository cleanup")


def validate_policy_source(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"step", "reset"} and isinstance(node.func.value, ast.Name) and node.func.value.id == "env":
                fail(f"Forbidden env call in {path}: line {node.lineno}")


def run_course_validators() -> None:
    for name in FINAL_SUBMISSIONS:
        policy = SUBMISSIONS / name / "policy.py"
        validate_policy_source(policy)
        run([sys.executable, "policy_validation_tests.py", str(policy), "--max-seconds-per-call", "0.25"])
    print("OK all five course-supplied policy validators")


def full_holdout_validation() -> None:
    # Full official-style 200-episode evaluation for every final policy.
    helper = ROOT / ".final_holdout_validation_tmp.py"
    helper.write_text(
        """import importlib.util\nimport json\nimport sys\nfrom pathlib import Path\nfrom evaluation import evaluate_policy, summarise_overall\n\nROOT = Path(__file__).resolve().parent\nPOLICIES = {\n    'PPO': ROOT/'submissions/ppo/policy.py',\n    'DQN': ROOT/'submissions/dqn/policy.py',\n    'Neural SARSA': ROOT/'submissions/neural_sarsa/policy.py',\n    'A2C': ROOT/'submissions/a2c/policy.py',\n    'Double DQN': ROOT/'submissions/double_dqn/policy.py',\n}\nfor name, path in POLICIES.items():\n    spec = importlib.util.spec_from_file_location('policy_'+name.replace(' ','_'), path)\n    mod = importlib.util.module_from_spec(spec)\n    spec.loader.exec_module(mod)\n    per_episode, summary = evaluate_policy(mod.run_policy, progress=False)\n    overall = summarise_overall(per_episode)\n    print(f\"{name}: mean={overall['mean_cost']:.2f} std={overall['std_cost']:.2f} service={overall['mean_service_level']:.6f} episodes={overall['n_episodes']}\")\n    if overall['n_episodes'] != 200:\n        raise SystemExit(f\"{name}: expected 200 episodes\")\n""",
        encoding="utf-8",
    )
    try:
        run([sys.executable, str(helper)])
    finally:
        helper.unlink(missing_ok=True)
    print("OK full official-style 200-episode holdout executed for all five policies")


def main() -> int:
    print("=== IITM RL final release ===", flush=True)
    check_assigned_config()
    check_inputs()
    replace_a2c_model()
    validate_notebook()
    validate_readme()
    validate_submission_layout()
    cleanup()
    validate_submission_layout()
    run_course_validators()
    full_holdout_validation()
    print("\nFINAL RELEASE VALIDATION PASSED", flush=True)
    print("Next: inspect git diff, commit, push the branch, then open/merge the PR.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
