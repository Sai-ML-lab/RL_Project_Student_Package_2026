# Remaining RL Experiment Plan

## Frozen PPO candidate

The preferred PPO candidate is:

- PPO + Representation B
- 500,000 transitions
- learning rate 6e-4
- seed 20260727
- local mean cost 83,039.20
- local std 9,777.14
- service 0.992973

The existing `submissions/ppo/` public champion remains untouched.

## Important PPO correction

The frozen Rep-B policy must exactly mirror:

- inventory normalization: 200
- pipeline normalization: 100
- demand-history normalization: 100
- day normalization: 49
- reference lead times: [3, 2, 1]

A policy using different constants is not reproducible with the trained model.

## Priority 1: A2C + Rep-B

Smoke test:

```bash
python training_pipelines/training_scripts/train_a2c_rep_b_experiment.py \
  --timesteps 4096 \
  --seed 20260727 \
  --run-name smoke_test
```

Real run:

```bash
python training_pipelines/training_scripts/train_a2c_rep_b_experiment.py \
  --timesteps 500000 \
  --seed 20260727 \
  --run-name a2c_rep_b_500k
```

## Priority 2: Neural SARSA controlled new seed

The current SARSA algorithm already uses Representation B. The useful experiment is a new seed with the current public hyperparameters.

Smoke:

```bash
python training_pipelines/run_neural_sarsa_rep_b_experiment.py \
  --timesteps 4096 \
  --seed 20260826 \
  --run-name smoke_test
```

Real:

```bash
python training_pipelines/run_neural_sarsa_rep_b_experiment.py \
  --timesteps 500000 \
  --seed 20260826 \
  --learning-rate 0.0003 \
  --gamma 0.98 \
  --epsilon-end 0.03 \
  --run-name sarsa_rep_b_500k_seed20260826
```

## Priority 3: A3C

Smoke:

```bash
python training_pipelines/training_scripts/train_a3c_experiment.py \
  --timesteps 4096 \
  --workers 1 \
  --rollout-steps 20 \
  --seed 20260727 \
  --run-name smoke_test
```

Real:

```bash
python training_pipelines/training_scripts/train_a3c_experiment.py \
  --timesteps 500000 \
  --workers 4 \
  --rollout-steps 20 \
  --seed 20260727 \
  --run-name a3c_rep_b_500k
```

## Priority 4: Online neural-network Q-learning

This is a distinct technique from the submitted DQN/Double-DQN because it uses
online one-step Q-learning without replay or a target network.

Smoke:

```bash
python training_pipelines/training_scripts/train_neural_q_learning_experiment.py \
  --timesteps 4096 \
  --seed 20260728 \
  --run-name smoke_test
```

Real:

```bash
python training_pipelines/training_scripts/train_neural_q_learning_experiment.py \
  --timesteps 500000 \
  --seed 20260728 \
  --learning-rate 0.0002 \
  --gamma 0.98 \
  --run-name qlearn_rep_b_500k
```

## Selection rule

Every candidate is evaluated on the same 200-episode local holdout:

40 reserved seeds (900-939) x five scenario modes.

Use the scenario breakdown, not just mean cost. Prefer candidates that improve
mean cost without an extreme deterioration in random/shock/trend performance.

Never overwrite a public submission while experimenting.
