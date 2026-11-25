"""Unit tests for the shared models and the 2D simulator: python3 -m pytest src/amr_sim_core/test -q"""
import xml.dom.minidom

import numpy as np
import pytest

from amr_sim_core.faults import (ActuationModel, FaultSampler, HealthState, LidarFaultModel, OdometryModel,
                                 apply_fault)
from amr_sim_core.geometry import raycast, rect_aabb_distance, rect_circle_distance
from amr_sim_core.metrics import EpisodeMetrics
from amr_sim_core.robot import load_robot
from amr_sim_core.world import load_scenario, load_world


@pytest.fixture(scope="module")
def world():
    return load_world("warehouse_small")


@pytest.fixture(scope="module")
def robot():
    return load_robot("amr_w1")


# ------------------------------------------------------------------ world
def test_world_layout(world):
    kinds = [b.kind for b in world.boxes]
    assert kinds.count("wall") == 4
    assert kinds.count("rack") == 16          # 2 blocks x 4 rows x 2 segments (one gap per row)
    assert world.aabbs.shape[1] == 4


def test_graph_connected(world):
    g = world.graph
    seen, stack = {0}, [0]
    while stack:
        for j in g.neighbors[stack.pop()]:
            if j not in seen:
                seen.add(j)
                stack.append(j)
    assert len(seen) == len(g.nodes)
    assert np.all(world.clearance(g.nodes[:, 0], g.nodes[:, 1]) >= 0.6 - 1e-9)


def test_sample_start_goal(world):
    rng = np.random.default_rng(0)
    for _ in range(20):
        s, g, path = world.sample_start_goal(rng, 5.0, 15.0)
        assert world.is_free(s[0], s[1], 0.6) and world.is_free(g[0], g[1], 0.6)
        assert 5.0 <= world.graph.path_length(path) <= 15.0


def test_rasterize_matches_boxes(world):
    grid, (ox, oy) = world.rasterize(0.05)
    res = 0.05
    for b in world.boxes:
        col = int((b.cx - ox) / res)
        row = grid.shape[0] - 1 - int((b.cy - oy) / res)
        assert grid[row, col] == 1, b.name


def test_gazebo_sdf_well_formed(world):
    from amr_sim_core.gazebo_sdf import world_to_sdf
    sdf = world_to_sdf(world)
    doc = xml.dom.minidom.parseString(sdf)
    names = [m.getAttribute("name") for m in doc.getElementsByTagName("model")]
    assert sum(n.startswith("ped_") for n in names) == world.ped_spec["max"]
    assert all(b.name in names for b in world.boxes)


# ------------------------------------------------------------------ geometry
def test_raycast_box_and_circle():
    boxes = np.array([[2.0, -1.0, 3.0, 1.0]])
    r = raycast(0.0, 0.0, np.array([0.0, np.pi]), 10.0, boxes)
    assert r[0] == pytest.approx(2.0) and r[1] == pytest.approx(10.0)
    r = raycast(0.0, 0.0, np.array([0.0]), 10.0, None, np.array([[5.0, 0.0]]), np.array([0.5]))
    assert r[0] == pytest.approx(4.5)


def test_rect_distances():
    boxes = np.array([[1.0, -0.2, 1.4, 0.2]])
    assert rect_aabb_distance(0.0, 0.0, 0.0, 0.35, 0.25, boxes)[0] == pytest.approx(0.65)
    assert rect_aabb_distance(1.2, 0.0, 0.0, 0.35, 0.25, boxes)[0] == 0.0
    # crossing case: thin long box through a rotated robot, no corner inside the other shape
    thin = np.array([[-2.0, -0.02, 2.0, 0.02]])
    assert rect_aabb_distance(0.0, 0.0, np.pi / 2, 0.35, 0.25, thin)[0] == 0.0
    d = rect_circle_distance(0.0, 0.0, 0.0, 0.35, 0.25, np.array([[1.0, 0.0]]), np.array([0.25]))
    assert d[0] == pytest.approx(0.4)


# ------------------------------------------------------------------ faults
def test_wheel_gain_makes_robot_turn(robot):
    h = HealthState(wheel_gain_left=0.5)
    act = ActuationModel(robot, h)
    act.push(0.0, 0.5, 0.0)
    v, w = act.target(0.0)
    assert v < 0.5 and w > 0.0                 # slower left wheel -> turns left


def test_latency(robot):
    act = ActuationModel(robot, HealthState(cmd_latency=0.3, faults=["cmd_latency"]))
    act.push(0.0, 0.5, 0.0)
    assert act.target(0.2)[0] == 0.0
    assert act.target(0.31)[0] == pytest.approx(0.5)


def test_payload_and_torque_reduce_acceleration(robot):
    a_empty = ActuationModel(robot, payload_mass=0.0).accel_limits(0.0)[0]
    a_loaded = ActuationModel(robot, payload_mass=60.0).accel_limits(0.0)[0]
    a_fault = ActuationModel(robot, HealthState(torque_gain=0.5, faults=["torque_loss"]), 60.0).accel_limits(0.0)[0]
    assert a_empty == pytest.approx(robot.max_lin_acc)
    assert a_fault < a_loaded < a_empty


def test_fault_onset(robot):
    h = HealthState(wheel_gain_left=0.5, onset_time=10.0, faults=["wheel_gain"])
    assert np.allclose(h.to_vector(5.0), 0.0)
    assert h.to_vector(11.0)[0] == pytest.approx(0.5)


def test_lidar_faults(robot):
    angles = robot.lidar_angles()
    ranges = np.full(len(angles), 5.0)
    rng = np.random.default_rng(0)
    h = HealthState(lidar_sector_center=0.0, lidar_sector_width=np.pi / 2, faults=["lidar_sector"])
    out = LidarFaultModel(angles, h).apply(ranges, 0.0, rng, 12.0)
    assert np.isinf(out[np.abs(angles) < np.pi / 4 - 1e-6]).all()
    assert np.isfinite(out[np.abs(angles) > np.pi / 4 + 1e-6]).all()
    h = HealthState(lidar_dropout_rate=0.5, faults=["lidar_dropout"])
    frac = np.isinf(LidarFaultModel(angles, h).apply(ranges, 0.0, rng, 12.0)).mean()
    assert 0.4 < frac < 0.6


def test_encoder_drift(robot):
    od = OdometryModel(robot, HealthState(encoder_scale_left=1.1, faults=["encoder_drift"]))
    od.reset([0, 0, 0])
    for _ in range(100):
        od.update(0.0, 0.1, 0.5, 0.0)
    # true motion is straight; the left wheel over-reads, so odometry believes the robot turned right
    assert od.pose[2] < -0.5 and np.hypot(od.pose[0], od.pose[1]) > 3.0


def test_fault_sampler_ranges():
    rng = np.random.default_rng(1)
    s = FaultSampler({"p_nominal": 0.0, "types": ["wheel_gain", "cmd_latency"], "severity": [0.6, 0.9],
                      "max_simultaneous": 2})
    for _ in range(50):
        h = s.sample(rng)
        assert 1 <= len(h.faults) <= 2
        assert all(0.6 <= x <= 0.9 for x in h.severities)
    with pytest.raises(ValueError):
        apply_fault(HealthState(), "teleport", 0.5, rng)


# ------------------------------------------------------------------ metrics
def test_near_miss_counting():
    m = EpisodeMetrics(0.5)
    m.reset(0.0, (0, 0), (10, 0))
    for k, d in enumerate([2.0, 0.4, 0.3, 0.45, 1.0, 0.3, -0.01]):
        m.update(0.1 * (k + 1), (0.1 * k, 0), [d], [7], 1.0)
    s = m.summary("x")
    assert s["near_miss_events"] == 2 and s["ped_collisions"] == 1 and m.first_collision == "collision_ped"


# ------------------------------------------------------------------ simulator / env
@pytest.mark.parametrize("scenario", ["nominal", "a1_train", "a1_eval_heldout", "a7_data", "a7_shift_humans"])
def test_scenarios_run(scenario):
    from amr_sim_core.sim2d import WarehouseSim2D
    sim = WarehouseSim2D(scenario=scenario, seed=0)
    s = sim.reset()
    for _ in range(30):
        s = sim.step(0.5, 0.1)
        if sim.done:
            break
    assert s["scan"].shape == (720,)
    assert load_scenario(scenario)["name"]


def test_gym_env_api():
    gym = pytest.importorskip("gymnasium")
    from gymnasium.utils.env_checker import check_env

    import amr_sim_core  # noqa: F401  (registers WarehouseNav-v0)
    env = gym.make("WarehouseNav-v0", scenario="a1_train", privileged=True).unwrapped
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=0)
    assert "health" in obs and info["health"].shape == (9,)
    for _ in range(50):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        if term or trunc:
            assert "episode_metrics" in info
            break
