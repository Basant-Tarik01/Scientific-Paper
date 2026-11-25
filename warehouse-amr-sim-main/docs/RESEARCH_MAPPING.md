# Mapping the proposals onto this repo

## A1: Health-conditioned RL navigation among humans

| proposal step | where it is in this repo | status |
|---|---|---|
| 1. Warehouse environment with aisles, cross-aisles, pedestrian behaviour, Nav2 baseline | `warehouse_small` world, `pedestrian_sim`, `nav2_sim.launch.py`, `episode_runner` | ready |
| 2. Fault injection at the right interfaces, validated | `fault_injector` node and `amr_sim_core.faults`; unit tests in `src/amr_sim_core/test` | ready (wheel gain, latency, dropout + 4 more) |
| 3. Teacher: PPO with privileged health | `WarehouseNav-v0` with `privileged=True`; `examples/train_ppo.py --privileged` | env ready, training to do |
| 4. Student + baselines (domain-randomised PPO, recurrent PPO) | `train_ppo.py` (no flag = DR-PPO, `--recurrent` = RecurrentPPO); `info["health"]` for distillation or an auxiliary loss | baselines ready, student to do |
| 5. Safety-filter baseline | `amr_sim_core.safety` has protective-field / TTC filters to start from; CBF filter to do | partial |
| 6. Evaluation: held-out severities, combined faults, densities, seeds | `a1_eval_heldout` scenario; `train_ppo.py --eval`; `evaluate_baselines_2d.py`; Gazebo `episode_runner` | ready |

Observation for the student: `scan`, `goal`, `velocity` (measured, affected by encoder faults),
`last_action`, `peds` (optional; on a real robot this needs a people tracker) and `payload`. Do **not**
include `health`. History comes from the recurrent encoder (or a frame-stack wrapper).

Suggested protocol: train in 2D (`a1_train`), evaluate in 2D on `a1_eval_heldout` with several seeds, then
transfer the best policies to Gazebo for a smaller validation set. The Gazebo evaluation is the
"second-simulator check" the proposal lists as optional. Wrap the policy as a node that subscribes `/scan`,
`/odom` and the goal, and publishes `/cmd_vel`.

## A7: JEPA-style LiDAR world model

| proposal step | where it is in this repo | status |
|---|---|---|
| 1. Generate data: 2D LiDAR, humans, commands, payload, health | `examples/generate_a7_dataset.py` (2D, fast) or `sim.launch.py log_data:=true` (Gazebo, `data_logger`); same `.npz` layout | ready |
| 2. Encode scans (grid or polar) | `amr_sim_core.representations.scan_to_bev / scan_to_polar`; `--bev` stores 64x64 grids | ready |
| 3. Baselines: constant velocity, ConvLSTM | data ready; models to do | to do |
| 4. Context-conditioned JEPA predictor | context per step: `cmd`, `payload`, `health` (and `odom_twist`) | to do |
| 5. Safety use: risk head, predictive slowdown vs fixed protective field and reactive TTC | `amr_sim_core.safety.ProtectiveField` (fixed and speed-dependent), `TTCRule`; `evaluate_baselines_2d.py` | baselines ready |
| 6. Generalisation and trust meter: faults, shifts | `a7_data` (fault onset during the episode), `a7_shift_humans`; add a second world YAML for "new layout" | partly ready |

Targets: the future scan at horizon k for step t is `scan[t + k]` in the same file (10 Hz). `scan_raw` is the
scan before sensor faults; `gt_pose` / `gt_twist` give ground-truth clearance and time-to-collision labels for
the risk head. A worker's future position comes from `peds[t + k]`.

## Shared design decisions to agree on

* near-miss threshold: 0.5 m surface-to-surface (in each scenario YAML)
* control period 0.1 s (10 Hz, the LiDAR rate)
* training vs held-out severity split: 0.6
* density levels 4 / 8 / 14 workers; 20 % distracted workers
