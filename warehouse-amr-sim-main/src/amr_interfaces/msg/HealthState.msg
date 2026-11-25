# True (privileged) robot health published by fault_injector. A deployable policy must NOT use it;
# it is for the A1 teacher, for labels/evaluation and for A7 context ablations.
std_msgs/Header header
float32 wheel_gain_left        # 1 = healthy, <1 = speed gain loss of the left wheel
float32 wheel_gain_right
float32 torque_gain            # 1 = healthy, scales the maximum drive force (accel/brake limit)
float32 cmd_latency            # s, extra command delay
float32 lidar_dropout_rate     # probability that a beam returns no echo
float32 lidar_sector_center    # rad, centre of the blind sector (robot frame)
float32 lidar_sector_width     # rad, width of the blind sector (0 = none)
float32 lidar_noise_std        # m, extra range noise
float32 encoder_scale_left     # 1 = healthy, odometry scale error of the left encoder
float32 encoder_scale_right
float32 payload_mass           # kg, carried load (context, not a fault)
float32 onset_time             # s (sim time since the fault profile was set) at which faults start
bool fault_active              # faults currently in effect
string[] faults                # active fault types
float32[] severities           # their severities in [0, 1]
float32[] vector               # normalised degradation vector (order: amr_sim_core.faults.HEALTH_VECTOR_NAMES)
