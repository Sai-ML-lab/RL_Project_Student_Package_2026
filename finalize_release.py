#!/usr/bin/env python3
"""Finalize the IITM RL course project release in a repeatable, fail-fast way."""

from __future__ import annotations

import json
import os
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

KEEP_TRAINING_ROOT = {
    TRAINING / "assigned_config.json",
    TRAINING / "src",
    TRAINING / "training_scripts",
    TRAINING / "training_utils",
}

KEEP_TRAINING_SCRIPTS = {
    "common.py",
    "train_ppo_rep_b_experiment.py",
    "train_dqn.py",
    "train_neural_sarsa_final.py",
    "train_a2c_rep_c.py",
    "train_double_dqn_experiment.py",
}

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

KEEP_RESULTS = {
    RESULTS / "final_results.csv",
    RESULTS / "ppo_learning_curve.png",
    RESULTS / "dqn_learning_curve.png",
    RESULTS / "a2c_learning_curve.png",
}

FINAL_POLICIES = [
    SUBMISSIONS / "ppo" / "policy.py",
    SUBMISSIONS / "dqn" / "policy.py",
    SUBMISSIONS / "neural_sarsa" / "policy.py",
    SUBMISSIONS / "a2c" / "policy.py",
    SUBMISSIONS / "double_dqn" / "policy.py",
]

REQUIRED_ARTIFACTS = [
    SUBMISSIONS / "ppo" / "model.zip",
    SUBMISSIONS / "dqn" / "model.zip",
    SUBMISSIONS / "neural_sarsa" / "policy_state.pt",
    SUBMISSIONS / "a2c" / "model.zip",
    SUBMISSIONS / "double_dqn" / "model.zip",
]


def fail(message: str) -> None:
    raise RuntimeError(message)


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def validate_inputs() -> None:
    if not ASSIGNED_CONFIG.exists():
        fail("Missing training_pipelines/assigned_config.json")
    config = json.loads(ASSIGNED_CONFIG.read_text(encoding="utf-8"))
    if config.get("variant_id") != "V030":
        fail(f"Unexpected variant_id: {config.get('variant_id')!r}")
    if config.get("config_fingerprint") != "5566e180ff842b10":
        fail(f"Unexpected config fingerprint: {config.get('config_fingerprint')!r}")
    print("OK assigned configuration")

    required = [NOTEBOOK, README, REPORT]
    required += [*FINAL_POLICIES, *REQUIRED_ARTIFACTS]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        fail("Missing required finalization inputs: " + ", ".join(missing))
    print("OK required finalization inputs")


def copy_final_a2c_model() -> None:
    if not A2C_SOURCE_MODEL.exists():
        fail(f"Missing A2C source model: {A2C_SOURCE_MODEL.relative_to(ROOT)}")
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
        ("94660.12", "94,660.12"),
        ("99566.57", "99,566.57"),
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
    expected = {
        "ppo": ("policy.py", "model.zip", "PPO"),
        "dqn": ("policy.py", "model.zip", "DQN"),
        "neural_sarsa": ("policy.py", "policy_state.pt", "Neural Network SARSA"),
        "a2c": ("policy.py", "model.zip", "A2C"),
        "double_dqn": ("policy.py", "model.zip", "Double DQN"),
    }
    for folder, (policy_name, model_name, label) in expected.items():
        policy = SUBMISSIONS / folder / policy_name
        model = SUBMISSIONS / folder / model_name
        metadata = SUBMISSIONS / folder / "metadata.json"
        if not policy.exists() or not model.exists() or not metadata.exists():
            fail(f"Incomplete submission bundle for {label}")
        meta = json.loads(metadata.read_text(encoding="utf-8"))
        if meta.get("technique") != label:
            fail(f"Wrong technique metadata for {label}: {meta.get('technique')!r}")
    print("OK final submission layout")


def validate_script_imports() -> None:
    # Compile all retained Python sources before destructive cleanup.
    sources = [
        p for p in KEEP_SOURCE_FILES
        if p.exists()
    ]
    sources += [
        TRAINING / "training_scripts" / name
        for name in KEEP_TRAINING_SCRIPTS
        if (TRAINING / "training_scripts" / name).exists()
    ]
    sources += FINAL_POLICIES
    for path in sources:
        run([sys.executable, "-m", "py_compile", str(path)])
    print("OK retained Python files compile")


def cleanup() -> None:
    # Delete experimental folders/files under training_pipelines/models, logs,
    # eval_results, tuning_results and other non-final material.
    for name in ["models", "logs", "eval_results", "tuning_results"]:
        target = TRAINING / name
        if target.exists():
            shutil.rmtree(target)

    # Remove stale candidate training scripts; retain only the five reproducible recipes.
    scripts_dir = TRAINING / "training_scripts"
    if scripts_dir.exists():
        for path in scripts_dir.iterdir():
            if path.name not in KEEP_TRAINING_SCRIPTS and path.is_file():
                path.unlink()

    # Retain only explicitly required source files plus package __init__.py files.
    src_dir = TRAINING / "src"
    if src_dir.exists():
        for path in sorted(src_dir.rglob("*"), reverse=True):
            if path.is_file():
                if path.name == "__init__.py" or path in KEEP_SOURCE_FILES:
                    continue
                path.unlink()
        for path in sorted(src_dir.rglob("*"), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()

    # The current training_utils directory is needed by the retained training scripts.
    # Remove only clearly exploratory leftovers if present.
    utils_dir = TRAINING / "training_utils"
    if utils_dir.exists():
        for path in utils_dir.rglob("*"):
            if path.is_file() and path.name.startswith("screen_"):
                path.unlink()

    # Retain only final learning curves/results evidence.
    if RESULTS.exists():
        for path in RESULTS.iterdir():
            if path.is_file() and path not in KEEP_RESULTS:
                path.unlink()

    # Remove Python caches generated by validation.
    for pycache in ROOT.rglob("__pycache__"):
        if pycache.is_dir():
            shutil.rmtree(pycache)
    for pyc in ROOT.rglob("*.pyc"):
        if pyc.is_file():
            pyc.unlink()

    print("OK cleanup completed")


def validate_policies() -> None:
    for policy in FINAL_POLICIES:
        run([
            sys.executable,
            "policy_validation_tests.py",
            str(policy.relative_to(ROOT)),
            "--max-seconds-per-call",
            "0.25",
        ])
    print("OK course policy validator passed for all five policies")


def validate_official_holdout() -> None:
    helper = ROOT / ".final_holdout_validation_tmp.py"
    helper.write_text(
        r'''from pathlib import Path
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training_pipelines"))
from evaluation import evaluate_policy, summarise_overall

policies = {
    "PPO": ROOT / "submissions/ppo/policy.py",
    "DQN": ROOT / "submissions/dqn/policy.py",
    "Neural Network SARSA": ROOT / "submissions/neural_sarsa/policy.py",
    "A2C": ROOT / "submissions/a2c/policy.py",
    "Double DQN": ROOT / "submissions/double_dqn/policy.py",
}
rows = []
for name, path in policies.items():
    details = evaluate_policy(str(path), max_episodes=200)
    summary = summarise_overall(details)
    summary["Technique"] = name
    rows.append(summary)

df = pd.DataFrame(rows)
if len(df) != 5 or not (df.get("n_episodes") == 200).all():
    raise RuntimeError("Official holdout did not evaluate exactly 200 episodes for all five policies")
print(df.to_string(index=False))
''',
        encoding="utf-8",
    )
    try:
        run([sys.executable, str(helper)])
    finally:
        helper.unlink(missing_ok=True)
    print("OK official 200-episode holdout validation passed")


def main() -> int:
    print("=== IITM RL final release ===")
    validate_inputs()
    copy_final_a2c_model()
    validate_text_files()
    validate_submission_layout()
    validate_script_imports()
    cleanup()
    validate_policies()
    validate_official_holdout()
    print("FINAL RELEASE VALIDATION PASSED")
    try:
        Path(__file__).unlink()
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
