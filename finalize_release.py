#!/usr/bin/env python3
"""Finalize the IITM RL course project release in a repeatable, fail-fast way."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAINING = ROOT / "training_pipelines"
MODELS = TRAINING / "models"
SUBMISSIONS = ROOT / "submissions"
RESULTS = ROOT / "results"
NOTEBOOK = ROOT / "RL_Project_Final.ipynb"
README = ROOT / "README.md"
ASSIGNED_CONFIG = TRAINING / "assigned_config.json"
REPORT = ROOT / "Project_Report_DA25M579_V030.docx"
A2C_SOURCE_MODEL = MODELS / "a2c_rep_c_experiments" / "screen_lr5e-04_g0.99_lam0.95_ent0.001_n64_vf0.5_shape_seed20260902" / "final_model.zip"
A2C_FINAL_MODEL = SUBMISSIONS / "a2c" / "model.zip"
KEEP_TRAINING_SCRIPTS = {"common.py", "train_ppo_rep_b_experiment.py", "train_dqn.py", "train_neural_sarsa_final.py", "train_a2c_rep_c.py", "train_double_dqn_experiment.py"}
KEEP_SOURCE_FILES = {
    TRAINING / "src" / "algorithms" / "neural_sarsa.py",
    TRAINING / "src" / "algorithms" / "common" / "checkpoint.py",
    TRAINING / "src" / "algorithms" / "common" / "env_factory.py",
    TRAINING / "src" / "algorithms" / "common" / "networks.py",
    TRAINING / "src" / "algorithms" / "common" / "replay_buffer.py",
    TRAINING / "src" / "algorithms" / "common" / "schedules.py",
    TRAINING / "src" / "environment" / "action_codec.py",
    TRAINING / "src" / "environment" / "wrappers.py",
    TRAINING / "src" / "features" / "normalizer.py",
    TRAINING / "src" / "features" / "observation.py",
    TRAINING / "src" / "features" / "engineered.py",
    TRAINING / "src" / "features" / "representation_c.py",
}
KEEP_UTILS = {"rep_c_env.py", "obs_wrapper.py", "action_wrapper.py", "reward_shaping.py", "env_factory.py", "__init__.py"}
KEEP_RESULTS = {RESULTS / "final_results.csv", RESULTS / "ppo_learning_curve.png", RESULTS / "dqn_learning_curve.png", RESULTS / "a2c_learning_curve.png"}
FINAL_POLICIES = [SUBMISSIONS / "ppo" / "policy.py", SUBMISSIONS / "dqn" / "policy.py", SUBMISSIONS / "neural_sarsa" / "policy.py", SUBMISSIONS / "a2c" / "policy.py", SUBMISSIONS / "double_dqn" / "policy.py"]
REQUIRED_ARTIFACTS = [SUBMISSIONS / "ppo" / "model.zip", SUBMISSIONS / "dqn" / "model.zip", SUBMISSIONS / "neural_sarsa" / "policy_state.pt", SUBMISSIONS / "a2c" / "model.zip", SUBMISSIONS / "double_dqn" / "model.zip"]
EXPECTED_METADATA = {
    "ppo": ("PPO", "Representation B", 76, 500000),
    "dqn": ("DQN", "35-dim hand-engineered features", 35, 150000),
    "neural_sarsa": ("Neural Network SARSA", "76-dim Representation B", 76, 500000),
    "a2c": ("A2C", "99-dimensional Representation C", 99, 500000),
    "double_dqn": ("Double DQN", "35-dimensional hand-engineered feature vector", 35, None),
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_inputs() -> None:
    if not ASSIGNED_CONFIG.exists():
        fail("Missing training_pipelines/assigned_config.json")
    config = json.loads(ASSIGNED_CONFIG.read_text(encoding="utf-8"))
    if config.get("variant_id") != "V030" or config.get("config_fingerprint") != "5566e180ff842b10":
        fail(f"Assigned configuration mismatch: {config.get('variant_id')!r}, {config.get('config_fingerprint')!r}")
    required = [NOTEBOOK, README, REPORT, *FINAL_POLICIES, *REQUIRED_ARTIFACTS]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        fail("Missing required finalization inputs: " + ", ".join(missing))
    print("OK assigned configuration")
    print("OK required finalization inputs")


def copy_final_a2c_model() -> None:
    if not A2C_SOURCE_MODEL.exists():
        fail(f"Missing A2C source model: {A2C_SOURCE_MODEL.relative_to(ROOT)}")
    source_hash = sha256(A2C_SOURCE_MODEL)
    A2C_FINAL_MODEL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(A2C_SOURCE_MODEL, A2C_FINAL_MODEL)
    if sha256(A2C_FINAL_MODEL) != source_hash:
        fail("A2C model SHA-256 mismatch after copy")
    print("OK A2C final model copied and hash-verified")


def _technique(meta: dict) -> str | None:
    return meta.get("technique") or meta.get("technique_category")


def _representation(meta: dict) -> str:
    return str(meta.get("representation") or meta.get("observation_representation") or "")


def _budget(meta: dict) -> int | None:
    for value in (
        meta.get("training_transitions"),
        meta.get("main_hyperparameters", {}).get("total_timesteps"),
        meta.get("main_hyperparameters", {}).get("total_transitions"),
        meta.get("hyperparameters", {}).get("total_timesteps"),
    ):
        if isinstance(value, int):
            return value
    return None


def validate_text_files() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if nb.get("nbformat") != 4:
        fail("Final notebook is not nbformat 4")
    text = "\n".join("".join(c.get("source", [])) for c in nb.get("cells", []))
    for alternatives in [
        ("Assigned parameter variant",),
        ("Reproducible final training pipelines",),
        ("Final public leaderboard comparison",),
        ("Final policy and model mapping", "Final policy/artifact mapping"),
        ("Course-supplied policy validation",),
        ("A2C + Representation C",),
        ("94660.12", "94,660.12"),
        ("99566.57", "99,566.57"),
    ]:
        if not any(x in text for x in alternatives):
            fail(f"Final notebook missing required text: {' or '.join(alternatives)}")
    for token in ["111,216.54", "111216.54", "114047.62", "Double-DQN pipeline use 76-dimensional", "public score pending"]:
        if token in text:
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
    expected_files = {
        "ppo": ("policy.py", "model.zip"),
        "dqn": ("policy.py", "model.zip"),
        "neural_sarsa": ("policy.py", "policy_state.pt"),
        "a2c": ("policy.py", "model.zip"),
        "double_dqn": ("policy.py", "model.zip"),
    }
    for folder, (policy_name, model_name) in expected_files.items():
        base = SUBMISSIONS / folder
        policy, model, metadata = base / policy_name, base / model_name, base / "metadata.json"
        if not all(p.exists() for p in (policy, model, metadata)):
            fail(f"Incomplete submission bundle for {folder}")
        meta = json.loads(metadata.read_text(encoding="utf-8"))
        label, rep_expected, dim, budget = EXPECTED_METADATA[folder]
        actual = _technique(meta)
        if actual != label:
            fail(f"Wrong technique metadata for {label}: {actual!r}")
        rep = _representation(meta)
        if rep_expected not in rep:
            fail(f"Wrong representation metadata for {label}: {rep!r}")
        if str(dim) not in rep:
            fail(f"Missing observation dimension {dim} for {label}: {rep!r}")
        if budget is not None and _budget(meta) != budget:
            fail(f"Wrong training budget for {label}: {_budget(meta)!r}; expected {budget}")
    print("OK final submission layout and metadata")


def validate_python_syntax(paths: list[Path]) -> None:
    for path in paths:
        if path.exists():
            run([sys.executable, "-m", "py_compile", str(path)])
    print("OK Python syntax checks")


def validate_training_utils() -> None:
    required = [TRAINING / "training_utils" / n for n in KEEP_UTILS if n != "__init__.py"]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        fail("Missing retained training utility dependencies: " + ", ".join(missing))
    validate_python_syntax(required)
    print("OK retained training utility dependencies")


def cleanup() -> None:
    for name in ["models", "logs", "eval_results", "tuning_results"]:
        target = TRAINING / name
        if target.exists():
            shutil.rmtree(target)
    scripts_dir = TRAINING / "training_scripts"
    if scripts_dir.exists():
        for p in scripts_dir.iterdir():
            if p.is_file() and p.name not in KEEP_TRAINING_SCRIPTS:
                p.unlink()
    src_dir = TRAINING / "src"
    if src_dir.exists():
        for p in sorted(src_dir.rglob("*"), reverse=True):
            if p.is_file() and p.name != "__init__.py" and p not in KEEP_SOURCE_FILES:
                p.unlink()
        for p in sorted(src_dir.rglob("*"), reverse=True):
            if p.is_dir() and not any(p.iterdir()):
                p.rmdir()
    utils_dir = TRAINING / "training_utils"
    if utils_dir.exists():
        for p in list(utils_dir.iterdir()):
            if p.is_file() and p.name not in KEEP_UTILS:
                p.unlink()
        for p in sorted(utils_dir.rglob("*"), reverse=True):
            if p.is_dir() and not any(p.iterdir()):
                p.rmdir()
    if RESULTS.exists():
        for p in RESULTS.iterdir():
            if p.is_file() and p not in KEEP_RESULTS:
                p.unlink()
    for base in [TRAINING, SUBMISSIONS, RESULTS]:
        for p in sorted(base.rglob("__pycache__"), reverse=True):
            if p.is_dir():
                shutil.rmtree(p)
    print("OK cleanup completed")


def validate_final_tree() -> None:
    expected_top = {"assigned_config.json", "src", "training_scripts", "training_utils"}
    actual_top = {p.name for p in TRAINING.iterdir()}
    if actual_top != expected_top:
        fail(f"Unexpected training_pipelines top-level contents: {sorted(actual_top)}")
    forbidden = {"models", "logs", "eval_results", "tuning_results", "a2c_rep_c_experiments", "a3c_rep_c_experiments", "ppo_rep_c_experiments", "dqn_rep_c_experiments", "ddqn_rep_c_experiments", "sarsa_experiments"}
    found = [str(p.relative_to(ROOT)) for p in TRAINING.rglob("*") if p.is_dir() and p.name in forbidden]
    if found:
        fail("Experimental directories remain: " + ", ".join(found))
    print("OK final tree contains no experimental directories")


def validate_policies() -> None:
    for p in FINAL_POLICIES:
        run([sys.executable, "policy_validation_tests.py", str(p.relative_to(ROOT)), "--max-seconds-per-call", "0.25"])
    print("OK course-supplied policy validator")


def validate_holdout() -> None:
    temp = ROOT / ".final_holdout_validation_tmp.py"
    temp.write_text(
        "from evaluation import evaluate_policy, summarise_overall\n"
        "from pathlib import Path\n"
        "policies=[Path('submissions/ppo/policy.py'),Path('submissions/dqn/policy.py'),Path('submissions/neural_sarsa/policy.py'),Path('submissions/a2c/policy.py'),Path('submissions/double_dqn/policy.py')]\n"
        "for policy in policies:\n"
        " print(f'=== HOLDOUT {policy.parent.name} ===')\n"
        " r=evaluate_policy(policy_path=policy,seeds=range(900,940),scenarios=['stationary','seasonal','trend','shock','random'])\n"
        " s=summarise_overall(r); print(s)\n"
        " if int(s['n_episodes']) != 200: raise SystemExit(f'Expected 200 episodes for {policy}, got {s[\"n_episodes\"]}')\n"
        "print('FULL_HOLDOUT_OK')\n",
        encoding="utf-8",
    )
    try:
        run([sys.executable, str(temp)])
    finally:
        temp.unlink(missing_ok=True)
    print("OK full 200-episode holdout for all five policies")


def main() -> int:
    print("=== IITM RL final release ===")
    validate_inputs()
    copy_final_a2c_model()
    validate_text_files()
    validate_submission_layout()
    pre_cleanup = [*FINAL_POLICIES, *KEEP_SOURCE_FILES, *(TRAINING / "training_scripts" / n for n in KEEP_TRAINING_SCRIPTS), *(TRAINING / "training_utils" / n for n in KEEP_UTILS)]
    validate_python_syntax(pre_cleanup)
    validate_training_utils()
    cleanup()
    validate_final_tree()
    post_cleanup = [*FINAL_POLICIES, *KEEP_SOURCE_FILES, *(TRAINING / "training_scripts" / n for n in KEEP_TRAINING_SCRIPTS), *(TRAINING / "training_utils" / n for n in KEEP_UTILS)]
    validate_python_syntax(post_cleanup)
    validate_policies()
    validate_holdout()
    print("FINAL RELEASE VALIDATION PASSED")
    try:
        (ROOT / "finalize_release.py").unlink()
        print("Removed finalize_release.py after successful finalization")
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
