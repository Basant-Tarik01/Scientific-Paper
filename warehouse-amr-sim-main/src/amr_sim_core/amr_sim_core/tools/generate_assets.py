"""Generate the Gazebo world, the Nav2 map and a preview image from a world YAML.

    python3 -m amr_sim_core.tools.generate_assets --world warehouse_small --out src/warehouse_world
    # or, after colcon build:
    ros2 run amr_sim_core generate_assets --world warehouse_small --out src/warehouse_world

Writes <out>/worlds/<name>.world, <out>/maps/<name>.pgm + .yaml and (with --preview) a PNG.
"""
from __future__ import annotations

import argparse
import os

from ..gazebo_sdf import world_to_sdf
from ..world import load_world


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", default="warehouse_small", help="bundled world name or path to a world YAML")
    ap.add_argument("--out", required=True, help="output package directory (e.g. src/warehouse_world)")
    ap.add_argument("--resolution", type=float, default=0.05, help="map resolution (m/cell)")
    ap.add_argument("--preview", default=None, help="optional PNG path for a top-down preview")
    args = ap.parse_args(argv)

    world = load_world(args.world)
    os.makedirs(os.path.join(args.out, "worlds"), exist_ok=True)
    wpath = os.path.join(args.out, "worlds", f"{world.name}.world")
    with open(wpath, "w") as f:
        f.write(world_to_sdf(world))
    mpath = world.write_nav2_map(os.path.join(args.out, "maps"), args.resolution)
    print(f"world : {wpath}\nmap   : {mpath}")
    print(f"boxes : {len(world.boxes)}  graph nodes: {len(world.graph.nodes)}")
    if args.preview:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from ..render import draw_world
        fig, ax = plt.subplots(figsize=(12, 8.4), dpi=110)
        draw_world(world, ax, show_graph=True, show_labels=True)
        sp = world.robot_spawn
        ax.plot(sp["x"], sp["y"], "ks", ms=8)
        ax.set_title(f"{world.name}: {world.width:.0f} x {world.height:.0f} m  (grey = pedestrian/route graph, "
                     f"square = robot spawn)")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        fig.tight_layout()
        fig.savefig(args.preview)
        print(f"preview: {args.preview}")


if __name__ == "__main__":
    main()
