"""amr_sim_core: shared world, robot, fault, pedestrian and metric models for the warehouse AMR
simulation (Gazebo/ROS 2 nodes and the fast 2D simulator / Gymnasium environment)."""
from .faults import FAULT_TYPES, HEALTH_VECTOR_NAMES, FaultSampler, HealthState, apply_fault  # noqa: F401
from .robot import load_robot  # noqa: F401
from .world import load_scenario, load_world  # noqa: F401

__version__ = "0.1.0"

try:  # register the Gymnasium environment if gymnasium is installed
    import gymnasium as _gym

    if "WarehouseNav-v0" not in _gym.registry:
        _gym.register(id="WarehouseNav-v0", entry_point="amr_sim_core.gym_env:WarehouseNavEnv")
except ImportError:  # pragma: no cover
    pass
