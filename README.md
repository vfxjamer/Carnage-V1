# Carnage V1: Phase 0 → Phase 1

Carnage is a fresh 1v1 Rocket League PPO training run built on the bundled GigaLearnCPP/RLGymCPP stack. It deliberately implements only Phase 0, the reviewed 50M-step transition, and Phase 1.

## Fixed compatibility contract

- NextoObs: 94 values
- DefaultAction: 90 discrete actions
- Policy and critic: `2048, 2048, 1024, 1024`, ReLU, LayerNorm
- 1 blue car versus 1 orange car
- `RandomState(true, true, false)`
- Tick skip 8; action delay 7 physics ticks
- PPO minibatch 25k, three epochs, gamma 0.99, lambda 0.95

Checkpoints with a different signature—including the former `1024, 1024, 512, 512` networks—are rejected. There is no partial or shape-converting load path.

## Curriculum

Phase 0 uses Touch 50, SpeedTowardBall 5, FaceBall 1, Air 0.15, no BallToGoal/Goal reward, `2e-4` policy and critic learning rates, and a 50k rollout/batch.

The first safe PPO boundary at or above nominal 200M saves `200M_phase0_final`, records the actual counter, marks the checkpoint as awaiting review, and exits successfully. Resume with `--approve-p0` only after capability review. Approval persists immediately and starts a full 50M-step transition from the actual restored counter.

The transition linearly reaches Touch 5, SpeedTowardBall 1, FaceBall 0.1, Air 0.15, BallToGoal 2, Goal/Concede ±20, and `1e-4` learning rates. Rollout/batch are 100k. Its first completed safe boundary publishes `250M_phase1_start` with nominal and actual counters.

Phase 1 remains at 100k until reviewed scoring is confirmed:

```bash
./Carnage --checkpoint-root checkpoints --resume auto \
  --scoring-confirmed --scoring-rollout 200000
```

`--scoring-rollout 300000` is the only other accepted larger size. Approval and rollout selection survive resume.

## Entropy and action latency

The bundled PPO source computes `H_raw = -Σ p ln(p)`, divides by `ln(action_count)` when `maskEntropy=false`, and applies `policy_loss - entropyScale × H_normalized`. Therefore this fixed 90-action run maps guide-equivalent `0.01` to:

```text
entropyScale = 0.01 × ln(90) = 0.044998096...
```

Both coefficients plus raw and normalized entropy are logged. A fixed conversion is rejected for per-sample valid-action normalization.

Action delay is measured in RocketSim physics ticks. The old control runs for seven ticks while policy inference uses the observation from the start of the interval; the new control runs for the eighth tick. Latency is `7/120 s = 58.33 ms`, with a `66.67 ms` decision interval. The RLBot client applies the queued control on tick 7 to match training.

## Checkpoints and termination

Every checkpoint is written to a temporary sibling directory, checked for all model/optimizer/statistics files, metadata schema, compatibility, sizes, and hashes, and renamed before `LATEST.json` is atomically replaced. Auto-resume validates candidates newest-first and falls back past invalid candidates.

On Windows, place `--checkpoint-root` outside OneDrive or another live-sync folder. Such software can hold newly serialized Torch files open and prevent the required atomic directory rename; the trainer fails closed instead of publishing a non-atomic checkpoint.

Rotating saves use threshold crossing at 5M intervals and retain eight rotating directories. Permanent checkpoints are never pruned. SIGINT/SIGTERM only request a stop; the learner completes the active PPO iteration, saves at its synchronized boundary, sends final metrics, and exits.

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --config Release -j
ctest --test-dir build -C Release --output-on-failure
```

Run locally:

```bash
./build/Carnage collision_meshes --device cuda --games 300 \
  --checkpoint-root checkpoints --resume auto --wandb carnage-v1
```

Use [Carnage_kaggle.ipynb](Carnage_kaggle.ipynb) for Kaggle. It reads `WANDB_API_KEY` from Kaggle Secrets, restores a private checkpoint dataset, validates both T4s, benchmarks single versus policy-T4:0/critic-T4:1 layouts and worker counts, and selects the fastest validated configuration. Dataset publishing is coalesced, final synchronization blocks, and `--delete-old-versions` prevents unbounded Kaggle dataset-version growth while retained milestone directories remain in the current version.
