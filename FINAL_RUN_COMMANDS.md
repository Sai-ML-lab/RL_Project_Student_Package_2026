# Final Remaining RL Experiment Run Plan

The code in this bundle has been statically validated with Python compileall.
The remaining training/evaluation runs should be executed in the project's
existing macOS `.venv`, because that environment contains the course's
Gymnasium/Stable-Baselines3 installation and the M4 hardware.

## 0. Verify frozen PPO candidate

python policy_validation_tests.py submissions/ppo_rep_b_500k/policy.py
python training_pipelines/tests/test_rep_b_inference.py

Do not modify `submissions/ppo/`.

## 1. A2C + Representation B

Smoke:
python training_pipelines/training_scripts/train_a2c_rep_b_experiment.py \
  --timesteps 4096 \
  --seed 20260727 \
  --run-name smoke_test

Real:
python training_pipelines/training_scripts/train_a2c_rep_b_experiment.py \
  --timesteps 500000 \
  --seed 20260727 \
  --run-name a2c_rep_b_500k

## 2. Neural SARSA controlled new seed

Smoke:
python training_pipelines/run_neural_sarsa_rep_b_experiment.py \
  --timesteps 4096 \
  --seed 20260826 \
  --run-name smoke_test

Real:
python training_pipelines/run_neural_sarsa_rep_b_experiment.py \
  --timesteps 500000 \
  --seed 20260826 \
  --learning-rate 0.0003 \
  --gamma 0.98 \
  --epsilon-end 0.03 \
  --run-name sarsa_rep_b_500k_seed20260826

## 3. A3C + Representation B

Smoke:
python training_pipelines/training_scripts/train_a3c_experiment.py \
  --timesteps 4096 \
  --workers 1 \
  --rollout-steps 20 \
  --seed 20260727 \
  --run-name smoke_test

Real:
python training_pipelines/training_scripts/train_a3c_experiment.py \
  --timesteps 500000 \
  --workers 4 \
  --rollout-steps 20 \
  --seed 20260727 \
  --run-name a3c_rep_b_500k

## 4. Online neural-network Q-learning + Representation B

Smoke:
python training_pipelines/training_scripts/train_neural_q_learning_experiment.py \
  --timesteps 4096 \
  --seed 20260728 \
  --run-name smoke_test

Real:
python training_pipelines/training_scripts/train_neural_q_learning_experiment.py \
  --timesteps 500000 \
  --seed 20260728 \
  --learning-rate 0.0002 \
  --gamma 0.98 \
  --run-name qlearn_rep_b_500k

## Selection rule

Each candidate is scored on the same 200-episode local holdout:
40 seeds (900-939) x 5 scenario modes.

Prefer a candidate only when the mean cost improves materially and there is
no severe deterioration in random/shock/trend performance.

Never overwrite an existing public submission while experimenting.
