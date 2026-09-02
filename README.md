# IITM RL Course Project 2026 — Final Release

**Course:** IITM Web M.Tech Program, Course ID 6002W  
**Project:** Industrial Inventory Control using Reinforcement Learning  
**Roll number:** DA25M579  
**Assigned variant:** V030

## Final five techniques

| Technique | Public mean cost |
|---|---:|
| PPO + Representation B | 83,610.75 |
| DQN | 90,670.62 |
| A2C + Representation C | 94,660.12 |
| Neural Network SARSA | 113,072.38 |
| Double DQN | 115,819.00 |

**Final public Top-5 average: 99,566.57**  
Lower cost is better.

## Final submission package

- `industrial_inventory_env/` — supplied official environment
- `submissions/` — five frozen policy files and model artifacts
- `RL_Project_Final.ipynb` — final course notebook covering reproducibility, validation and results
- `Project_Report_DA25M579_V030.docx` — concise final report
- `starter_notebook.ipynb` — professor-provided starter material, retained unchanged
- `requirements.txt` — permitted package versions
- `evaluation.py` — local official-cost evaluation harness
- `policy_validation_tests.py` — course-supplied policy-interface validator
- `training_pipelines/` — source and launchers required to reproduce the final techniques

## Final A2C replacement

The protected A2C artifact was replaced by the Representation C candidate. Its official 200-episode holdout mean was 94,180.86 with mean service 0.994717, followed by a public score of 94,660.12. The final policy is `submissions/a2c/policy.py` and loads `submissions/a2c/model.zip`.

## Validation requirements

Before submission, run the course-supplied validator against all five final policy files. The validator checks importability, the exact `run_policy(observation)` interface, valid actions, deterministic inference, observation immutability, prohibited direct environment calls, and local inference-time limits.

The policy contract is inference-only: no training, environment stepping/resetting, internet access, future demand, scenario identity or hidden parameters are used by `run_policy()`.

## Reproducibility and cleanup

The official environment, official cost function and leaderboard evaluator are unchanged. The assigned V030 configuration is frozen in `training_pipelines/assigned_config.json`. Failed exploratory checkpoints and stale tuning artifacts are not part of the final deliverable; their material outcomes are summarized in the final notebook/report where relevant.
