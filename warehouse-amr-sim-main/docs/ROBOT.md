# AMR-W1 robot

The robot is a differential-drive warehouse AMR with a flat payload deck. The parameters live in
`src/amr_sim_core/amr_sim_core/data/robot/amr_w1.yaml`. The URDF (`amr_description/urdf/amr_w1.urdf.xacro`)
and the 2D simulator both read this file, so a change there affects both simulators.

| item | value |
|---|---|
| footprint | 0.70 m x 0.50 m (rectangle, centred on `base_footprint`) |
| chassis | 0.05-0.22 m above floor, 45 kg |
| payload deck | plate at 0.30-0.34 m, 4 kg. Optional payload box 0.6 x 0.4 x 0.3 m, 0-60 kg |
| drive | 2 wheels, r = 0.08 m, separation 0.44 m, on the centre axle. Front and rear frictionless casters |
| limits | v in [-0.3, 1.0] m/s, w in [-1.5, 1.5] rad/s, a <= 1.2 m/s^2, alpha <= 3.0 rad/s^2 |
| motor | 3 N m peak per wheel. Limits acceleration when loaded (see below) |
| LiDAR | 360 deg, 720 beams (0.5 deg), 10 Hz, 0.12-12 m, sigma = 1 cm, at z = 0.26 m in the slot between chassis and deck |
| IMU | 100 Hz, at the chassis |

## Frames

```
map (= Gazebo world frame) -> odom -> base_footprint -> base_link -> lidar_link, imu_link, deck_link [-> payload_link],
                                                                     left/right_wheel_link, front/rear_caster_link
```

* `map -> odom` comes from AMCL (Nav2) or, without Nav2, from `gt_localization` (perfect localisation from ground truth,
  `sim.launch.py`, `static_map_tf:=true`).
* `odom -> base_footprint` comes from `fault_injector`, using the odometry after encoder faults.
* `/ground_truth/odom` gives the true pose in the world frame. The world frame equals the map frame because the
  Nav2 map is generated from the same layout.

## Dynamics model (same in both simulators)

Commanded velocities pass through `amr_sim_core.faults.ActuationModel`:

1. **Latency FIFO.** The command takes effect after `cmd_latency` seconds.
2. **Velocity limits.**
3. **Per-wheel gain.** `v_l, v_r = v -/+ w*b/2`, each scaled by `wheel_gain_left/right`, then converted back to (v, w).
4. **Drive-force-limited acceleration.**
   `a_max = min(1.2, torque_gain * 2 * tau_max / r / (m_robot + m_payload))`.
   An empty robot reaches the nominal 1.2 m/s^2. With 60 kg the limit is 0.71 m/s^2, and with 60 kg plus 50 %
   torque loss it is 0.36 m/s^2. Braking distance grows accordingly. A7 needs this effect ("the same command
   gives different braking with an empty or loaded platform").

The 2D simulator adds a first-order actuator lag (tau = 0.05 s). In Gazebo the diff-drive plugin and the
ODE physics play that role.

Inspect the model: `ros2 launch amr_description display.launch.py payload_mass:=40`.
