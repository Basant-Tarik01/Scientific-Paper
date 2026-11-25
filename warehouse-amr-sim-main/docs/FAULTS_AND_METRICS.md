# Faults, health vector and metrics

## Fault interfaces

Every fault is injected where it would occur on a real robot, as A1 step 2 requires. In Gazebo this happens
in the `fault_injector` node; in the 2D simulator it happens in the same classes (`amr_sim_core/faults.py`).

| type | parameter(s) | interface | severity s in [0, 1] |
|---|---|---|---|
| `wheel_gain` | `wheel_gain_left` / `_right` | actuation: per-wheel speed gain | one random wheel: gain = 1 - 0.7 s |
| `torque_loss` | `torque_gain` | actuation: max drive force, so acceleration and braking | torque_gain = 1 - 0.8 s |
| `cmd_latency` | `cmd_latency` | command processing: FIFO delay | latency = 0.5 s x s |
| `lidar_dropout` | `lidar_dropout_rate` | LiDAR stream: random beams return `inf` | rate = 0.6 s |
| `lidar_sector` | `lidar_sector_center/_width` | LiDAR stream: contiguous blind sector (dirt, occlusion) | width = pi x s (up to 180 deg), random centre |
| `lidar_noise` | `lidar_noise_std` | LiDAR stream: extra Gaussian range noise | std = 0.15 m x s |
| `encoder_drift` | `encoder_scale_left/_right` | odometry: wheel-encoder scale error | one wheel: 1 +/- 0.2 s |

* **Onset.** A fault can start partway through an episode (`faults.onset` in the scenario, or the
  `onset_time` parameter). Before the onset the robot is nominal. A7 uses this to test the trust meter
  around the moment a fault appears.
* **Combined faults.** `faults.max_simultaneous` > 1 samples several distinct fault types at once.
* **Train/held-out split.** The split is controlled by the `faults.severity` range. `a1_train` uses
  [0, 0.6] and `a1_eval_heldout` uses [0.6, 0.9].
* **No return.** Lost LiDAR returns are `inf`, as a real scanner reports them. Nav2 is configured with
  `inf_is_valid: true`, so a dropped beam *clears* the costmap. That is the realistic, dangerous effect.

## Health vector (privileged)

`HealthState.to_vector(t)` returns 9 values, each 0 when healthy. Gazebo publishes the same values on
`/amr/health.vector`, and the 2D env exposes them as `obs["health"]` (teacher only) and `info["health"]`:

| idx | name | definition |
|---|---|---|
| 0 | wheel_loss_left | 1 - wheel_gain_left |
| 1 | wheel_loss_right | 1 - wheel_gain_right |
| 2 | torque_loss | 1 - torque_gain |
| 3 | cmd_latency | latency / 0.5 s |
| 4 | lidar_dropout | dropout rate |
| 5 | lidar_sector | blind width / 2 pi |
| 6 | lidar_noise | extra std / 0.15 m |
| 7 | encoder_err_left | (scale - 1) / 0.2, in [-1, 1] |
| 8 | encoder_err_right | (scale - 1) / 0.2, in [-1, 1] |

Payload mass is *context*, not a fault. It is a separate observation (`obs["payload"]`, `/amr/health.payload_mass`).

## Metrics (fixed in advance)

`amr_sim_core/metrics.py` defines the metrics. Gazebo (`metrics_monitor`, using ground truth) and the 2D
simulator use the same code:

| metric | definition |
|---|---|
| distance | surface to surface: robot footprint rectangle to worker body circle, or to obstacle boxes |
| `collision_ped` / `collision_static` | distance <= 0. The first collision decides the episode outcome |
| `collision_speed` | robot speed at the first collision (about 0 means a worker walked into a stopped robot) |
| `near_miss_events` | a worker enters 0 < d < **0.5 m** (`metrics.near_miss_distance`). Counted once per worker per entry, with 0.1 m hysteresis |
| `near_miss_time` | time with at least one worker inside the near-miss band |
| `min_ped_distance`, `min_obstacle_distance` | minima over the episode |
| `success` | goal within 0.3 m without any collision before `episode.max_time` |
| `time`, `path_length`, `mean_speed` | efficiency |

Episode outcomes: `success`, `collision_ped`, `collision_static`, `timeout`. The Gazebo `episode_runner`
also reports `nav_failed` (Nav2 aborted) and `nav2_result`.
