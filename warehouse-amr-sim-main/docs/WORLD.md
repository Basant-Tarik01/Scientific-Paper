# Warehouse world

`src/amr_sim_core/amr_sim_core/data/worlds/warehouse_small.yaml` defines the warehouse. Everything else
is generated from it:

```bash
python3 -m amr_sim_core.tools.generate_assets --world warehouse_small --out src/warehouse_world \
        --preview docs/img/warehouse_small_layout.png
colcon build --packages-select warehouse_world amr_sim_core
```

This writes `warehouse_world/worlds/<name>.world` (Gazebo SDF), `warehouse_world/maps/<name>.pgm/.yaml`
(Nav2 map, 5 cm) and a preview PNG. The collision geometry is the same set of axis-aligned boxes in Gazebo,
in the Nav2 map and in the 2D simulator. Rack uprights, beams, cartons, pallet wrap, floor markings and worker
models are visual only.

## Layout of `warehouse_small` (30 m x 20 m)

* **Two rack blocks (A, B)**, each with 4 double-deep rack rows (9 m long, 1.2 m deep, 2.4 m high).
  The rows are separated by **2.4 m picking aisles**.
* A **2 m narrow cross-aisle** cuts through the middle of every row. These cross-aisles create *blind
  intersections* where workers appear from behind racks.
* A **4 m central cross-aisle** between the blocks, with two building pillars that cause occlusions.
* A **main transport aisle** along the bottom wall (y < 3 m) and 4 m side aisles.
* **Packing stations** along the top, a **pallet dock** in the top-right corner and a **charging dock**
  bottom-left, where the robot spawns.
* Yellow floor lines in front of every rack face and green outlines around the dock and charging zones (visual only).

## Waypoint graph

The world builds a graph automatically: nodes on a 1 m grid with at least 0.6 m clearance, joined by
collision-free edges. It is used to:

* route workers (shortest path, with lateral jitter so they do not walk in single file),
* sample robot start/goal pairs with path-length limits (`episode.min_goal_distance / max_goal_distance`),
* plan the global route of the scripted baseline controller.

## Workers (pedestrians)

`amr_sim_core.pedestrians.Crowd` implements a social-force model (Helbing). It includes anisotropic
person-person repulsion, wall repulsion, robot repulsion that anticipates the robot's motion by 0.5 s, and a
hard constraint that keeps workers out of obstacles. Each worker repeatedly:

1. picks a goal, biased by `pedestrians.zone_goal_weights` (picking aisles, packing, dock),
2. walks there at its own preferred speed (scenario `speed` range),
3. dwells (picking or packing) with probability `dwell_prob` for `dwell_time` seconds, and
4. re-plans if it gets stuck for 6 s.

A fraction `distracted_fraction` of workers **ignore the robot**: they do not yield, and they are the main
safety stress test. RViz shows aware workers in orange and distracted workers in red.

Density levels are set in the world YAML: none = 0, low = 4, medium = 8, high = 14. The Gazebo world contains
16 worker models (`ped_0..15`); inactive workers wait outside the building, beyond LiDAR range.

## Adding a world

Copy `warehouse_small.yaml` and change `name`, `size`, `rack_blocks`, `obstacles`, `zones` and
`robot_spawn`. Then run `generate_assets` and pass `world:=<name>` to the launch files or `world=` to
`WarehouseSim2D`. A second layout (for example a different aisle width) is the natural "new layout"
distribution shift for the A7 trust meter.
