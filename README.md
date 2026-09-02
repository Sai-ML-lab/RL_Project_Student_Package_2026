# IITM RL Course Project 2026 — Final Release

**Course:** IITM Web M.Tech Program, Course ID 6002W
**Project:** Industrial Inventory Control using Reinforcement Learning
**Roll number:** DA25M579
**Assigned variant:** V030

## Final five techniques

1. PPO + Representation B — public score 83,610.75
2. DQN — public score 90,670.62
3. Neural Network SARSA — public score 113,072.38
4. A2C — public score 114,047.62
5. Double DQN — current public score 115,819.00; final portal candidate is the new
   dueling joint-action Double DQN model with local 200-episode mean cost 111,216.54

Lower cost is better.

## Final project contents

- `industrial_inventory_env/` — supplied official environment
- `training_pipelines/` — only the source/launchers needed to reproduce the final five
- `submissions/` — five frozen policy + model artifact directories
- `portal_uploads/` — five upload-ready policy packages
- `RL_Project_Final.ipynb` — final notebook covering course-required reproducibility/evidence
- `Project_Report_DA25M579_V030.docx` — concise final report
- `evaluation.py` — local official-cost evaluation harness
- `policy_validation_tests.py` — supplied policy-interface validator
- `COURSE_COMPLIANCE_CHECKLIST.md` — final requirements checklist
- `EXPERIMENT_HISTORY.md` — text-only record of material rejected experiments

## Validation

The custom Neural Network SARSA and Double DQN policies were runtime-validated in this
release environment. All five final policy files were also statically checked for the required
`run_policy(observation)` interface, valid model artifact paths, and absence of direct
`env.step()`/`env.reset()` calls. PPO/DQN/A2C runtime imports require the same Python environment
used on the student's Mac because Stable-Baselines3 is not installed in the artifact-building
container.

Before portal upload, run the course-supplied validator locally against all five policy files.

## Important course note

The professor-provided `starter_notebook.ipynb` is retained in the final repository unchanged as supplied course material. It provides the approved configuration-generation workflow for the assigned V030 variant; the resulting configuration is frozen in `training_pipelines/assigned_config.json`.

The final code tree intentionally excludes failed/stale exploratory checkpoints and scripts. Their material outcomes are documented in the final notebook/report rather than retained as executable experiment artifacts.
