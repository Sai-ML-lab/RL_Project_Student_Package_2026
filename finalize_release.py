#!/usr/bin/env python3
"""Fail-fast final release validator for the IITM RL course project."""
from __future__ import annotations

import hashlib
import json
import re
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
REPORT = ROOT / "Project_Report_DA25M579_V030.docx"
CONFIG = TRAINING / "assigned_config.json"

A2C_SOURCE = MODELS / "a2c_rep_c_experiments" / "screen_lr5e-04_g0.99_lam0.95_ent0.001_n64_vf0.5_shape_seed20260902" / "final_model.zip"
A2C_TARGET = SUBMISSIONS / "a2c" / "model.zip"

FINAL = {
    "ppo": {"policy": "policy.py", "model": "model.zip", "technique": {"PPO"}, "dim": 76, "budget": 500000},
    "dqn": {"policy": "policy.py", "model": "model.zip", "technique": {"DQN", "Deep Q-Network (DQN)"}, "dim": 35, "budget": 150000},
    "neural_sarsa": {"policy": "policy.py", "model": "policy_state.pt", "technique": {"Neural Network SARSA"}, "dim": 76, "budget": 500000},
    "a2c": {"policy": "policy.py", "model": "model.zip", "technique": {"A2C", "Advantage Actor-Critic (A2C)"}, "dim": 99, "budget": 500000},
    "double_dqn": {"policy": "policy.py", "model": "model.zip", "technique": {"Double DQN"}, "dim": 35, "budget": None},
}

KEEP_SCRIPTS = {
    "common.py",
    "train_ppo_rep_b_experiment.py",
    "train_dqn.py",
    "train_neural_sarsa_final.py",
    "train_a2c_rep_c.py",
    "train_double_dqn_experiment.py",
}
KEEP_UTILS = {"__init__.py", "rep_c_env.py", "obs_wrapper.py", "action_wrapper.py", "reward_shaping.py", "env_factory.py"}
KEEP_SOURCE = {
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
POLICIES = [SUBMISSIONS / name / "policy.py" for name in FINAL]
ARTIFACTS = [SUBMISSIONS / name / spec["model"] for name, spec in FINAL.items()]


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metadata_technique(meta: dict) -> str | None:
    return meta.get("technique") or meta.get("technique_category")


def metadata_representation(meta: dict) -> str:
    return str(meta.get("representation") or meta.get("observation_representation") or "")


def metadata_dimension(meta: dict, rep: str) -> int | None:
    value = meta.get("observation_dimension")
    if isinstance(value, int):
        return value
    match = re.search(r"(\d{2,3})[- ]?(?:dim|dimension)", rep, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def metadata_budget(meta: dict) -> int | None:
    candidates = [
        meta.get("training_transitions"),
        meta.get("main_hyperparameters", {}).get("total_timesteps"),
        meta.get("main_hyperparameters", {}).get("total_transitions"),
        meta.get("hyperparameters", {}).get("total_timesteps"),
    ]
    return next((x for x in candidates if isinstance(x, int)), None)


def validate_inputs() -> None:
    if not CONFIG.exists():
        fail("Missing assigned_config.json")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("variant_id") != "V030" or config.get("config_fingerprint") != "5566e180ff842b10":
        fail(f"Assigned configuration mismatch: {config.get('variant_id')!r}, {config.get('config_fingerprint')!r}")
    required = [NOTEBOOK, README, REPORT, *POLICIES, *ARTIFACTS]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        fail("Missing required finalization inputs: " + ", ".join(missing))
    print("OK assigned configuration")
    print("OK required finalization inputs")


def copy_a2c() -> None:
    if not A2C_SOURCE.exists():
        fail(f"Missing A2C source model: {A2C_SOURCE.relative_to(ROOT)}")
    source_hash = sha256(A2C_SOURCE)
    A2C_TARGET.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(A2C_SOURCE, A2C_TARGET)
    if sha256(A2C_TARGET) != source_hash:
        fail("A2C model SHA-256 mismatch")
    print("OK A2C final model copied and hash-verified")


def validate_documents() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join("".join(c.get("source", [])) for c in nb.get("cells", []))
    for token in ["Assigned parameter variant", "Reproducible final training pipelines", "Final public leaderboard comparison", "Final policy and model mapping", "Course-supplied policy validation", "A2C + Representation C"]:
        if token not in text:
            fail(f"Final notebook missing required text: {token}")
    for alternatives in [("94660.12", "94,660.12"), ("99566.57", "99,566.57")]:
        if not any(x in text for x in alternatives):
            fail(f"Final notebook missing required result: {' / '.join(alternatives)}")
    for token in ["111,216.54", "111216.54", "114047.62", "public score pending"]:
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


def validate_metadata() -> None:
    for folder, spec in FINAL.items():
        base = SUBMISSIONS / folder
        meta_path = base / "metadata.json"
        if not meta_path.exists():
            fail(f"Missing metadata.json for {folder}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        technique = metadata_technique(meta)
        if technique not in spec["technique"]:
            fail(f"Wrong technique metadata for {folder}: {technique!r}")
        rep = metadata_representation(meta)
        if not rep:
            fail(f"Missing representation metadata for {folder}")
        dim = metadata_dimension(meta, rep)
        if dim != spec["dim"]:
            fail(f"Wrong observation dimension for {folder}: {dim!r}; expected {spec['dim']}; representation={rep!r}")
        budget = spec["budget"]
        if budget is not None and metadata_budget(meta) != budget:
            fail(f"Wrong training budget for {folder}: {metadata_budget(meta)!r}; expected {budget}")
    print("OK final submission layout and metadata")


def compile_paths(paths: list[Path], label: str) -> None:
    for path in paths:
        if path.exists():
            run([sys.executable, "-m", "py_compile", str(path)])
    print(f"OK {label}")


def validate_pre_cleanup_code() -> None:
    paths = POLICIES[:]
    paths += [TRAINING / "src" / rel for rel in KEEP_SOURCE]
    paths += [TRAINING / "training_scripts" / name for name in KEEP_SCRIPTS]
    paths += [TRAINING / "training_utils" / name for name in KEEP_UTILS]
    compile_paths(paths, "retained Python files compile before cleanup")


def cleanup() -> None:
    for name in ["models", "logs", "eval_results", "tuning_results"]:
        target = TRAINING / name
        if target.exists():
            shutil.rmtree(target)

    scripts = TRAINING / "training_scripts"
    if scripts.exists():
        for p in scripts.iterdir():
            if p.is_file() and p.name not in KEEP_SCRIPTS:
                p.unlink()

    src = TRAINING / "src"
    if src.exists():
        for p in src.rglob("*.py"):
            rel = p.relative_to(src).as_posix()
            if p.name != "__init__.py" and rel not in KEEP_SOURCE:
                p.unlink()
        for p in sorted(src.rglob("*"), reverse=True):
            if p.is_dir() and not any(p.iterdir()):
                p.rmdir()

    utils = TRAINING / "training_utils"
    if utils.exists():
        for p in utils.iterdir():
            if p.is_file() and p.name not in KEEP_UTILS:
                p.unlink()
        for p in sorted(utils.rglob("*"), reverse=True):
            if p.is_dir() and not any(p.iterdir()):
                p.rmdir()

    if RESULTS.exists():
        for p in RESULTS.iterdir():
            if p.is_file() and p.name not in KEEP_RESULTS:
                p.unlink()

    for base in [TRAINING, SUBMISSIONS, RESULTS]:
        for cache in base.rglob("__pycache__"):
            if cache.is_dir():
                shutil.rmtree(cache)
    print("OK cleanup completed")


def validate_tree() -> None:
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
    for policy in POLICIES:
        run([sys.executable, "policy_validation_tests.py", str(policy.relative_to(ROOT)), "--max-seconds-per-call", "0.25"])
    print("OK course-supplied policy validator")


def validate_holdout() -> None:
    script = ROOT / ".final_holdout_validation_tmp.py"
    script.write_text(
        "from evaluation import evaluate_policy, summarise_overall\n"
        "from pathlib import Path\n"
        "policies=[Path('submissions/ppo/policy.py'),Path('submissions/dqn/policy.py'),Path('submissions/neural_sarsa/policy.py'),Path('submissions/a2c/policy.py'),Path('submissions/double_dqn/policy.py')]\n"
        "for p in policies:\n"
        " print(f'=== HOLDOUT {p.parent.name} ===')\n"
        " r=evaluate_policy(policy_path=p,seeds=range(900,940),scenario_modes=['stationary','seasonal','trend','shock','random'])\n"
        " s=summarise_overall(r)\n"
        " print(s)\n"
        " assert int(s['n_episodes']) == 200, s\n"
        "print('FULL_HOLDOUT_OK')\n",
        encoding="utf-8",
    )
    try:
        run([sys.executable, str(script)])
    finally:
        script.unlink(missing_ok=True)
    print("OK full 200-episode holdout for all five policies")


def main() -> int:
    print("=== IITM RL final release ===")
    validate_inputs()
    copy_a2c()
    validate_documents()
    validate_metadata()
    validate_pre_cleanup_code()
    cleanup()
    validate_tree()
    post = POLICIES[:]
    post += [TRAINING / "src" / rel for rel in KEEP_SOURCE]
    post += [TRAINING / "training_scripts" / name for name in KEEP_SCRIPTS]
    post += [TRAINING / "training_utils" / name for name in KEEP_UTILS]
    compile_paths(post, "retained Python files compile after cleanup")
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
