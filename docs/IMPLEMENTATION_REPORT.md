# Phase 0 → Phase 1 Implementation Report

## Entropy

```text
Guide entropy coefficient:
0.01

Actual bundled GigaLearn entropy formula:
H_raw = mean(-sum(p * ln(p)))
H_normalized = H_raw / ln(action_count) when maskEntropy=false
PPO loss = (policy_loss - configured_runtime_coef * H_normalized) * batch_size_ratio

Configured GigaLearn runtime coefficient:
0.01 * ln(90) = 0.044998096...

Reason/conversion:
DefaultAction has 90 actions and this run requires maskEntropy=false. Multiplication by ln(90)
exactly cancels the bundled normalization and preserves a raw guide-equivalent coefficient of 0.01.
```

## Action delay

```text
Action delay configured:
7

Exact implementation semantics:
The observation is captured at the start of an eight-tick environment step. RocketSim advances
seven physics ticks under the preceding controls while inference runs. The inferred control is
then installed and RocketSim advances the remaining one tick. No observation or stored-action
queue implements this delay.

Effective latency with tick skip 8:
7 / 120 seconds = 58.33 ms observation-to-action latency; 8 / 120 seconds = 66.67 ms cadence.

Preserved or changed:
The configured trainer value remains 7. The RLBot client's former actionDelay - 1 comparison was
corrected so live inference now applies the queued action on tick 7.

Reason:
Seven ticks is intentional inference/simulation overlap in the bundled architecture. Correcting
the pre-training client parity bug avoids deploying a six-tick latency without changing training.
```

## Curriculum checkpoint reporting

Checkpoint metadata always contains `nominal_milestone` and `total_timesteps`. Actual values are only known when training crosses each boundary and are printed on publication. Expected records are:

| Checkpoint | Nominal target | Actual saved timestep |
|---|---:|---:|
| `200M_phase0_final` | 200,000,000 | First completed PPO boundary ≥ target |
| `250M_phase1_start` | 250,000,000 | First boundary ≥ actual approval step + 50,000,000 |
| `500M_milestone` | 500,000,000 | First completed PPO boundary ≥ target |
| `1B_milestone` | 1,000,000,000 | First completed PPO boundary ≥ target |
| `1_5B_milestone` | 1,500,000,000 | First completed PPO boundary ≥ target |
| `2B_final` | 2,000,000,000 | First completed PPO boundary ≥ target |

No training run has been executed far enough during implementation to invent actual milestone counters.
