#!/usr/bin/env python3
"""Generate a LiDAR world-model dataset (A7) in the fast 2D simulator.

One .npz per episode with the same keys as the Gazebo `data_logger` node, so training code can use
both sources. Because the simulator keeps running, future scans are simply later rows: the target
for horizon k at step t is scan[t + k].

    python3 examples/generate_a7_dataset.py --episodes 200 --out ~/amr_datasets/a7_2d --scenario a7_data
    python3 examples/generate_a7_dataset.py --episodes 50 --out ~/amr_datasets/a7_shift --scenario a7_shift_humans

Keys: t, scan, scan_raw (scan before sensor faults), angles, cmd (policy v, w), cmd_hw (after actuation faults),
odom_pose, odom_twist, gt_pose, gt_twist (vx, vy, w), health (9), payload, peds (K x 7), meta (json),
plus bev (T x 64 x 64, robot-centred occupancy, optional --bev) for convenience.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "amr_sim_core"))
from amr_sim_core.controllers import PathFollower  # noqa: E402
from amr_sim_core.representations import scan_to_bev  # noqa: E402
from amr_sim_core.safety import ProtectiveField  # noqa: E402
from amr_sim_core.sim2d import WarehouseSim2D  # noqa: E402

K_PEDS = 16


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="a7_data")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="datasets/a7_2d")
    ap.add_argument("--no-safety", action="store_true", help="drive without the protective field (more near-misses)")
    ap.add_argument("--bev", action="store_true", help="also store 64x64 bird's-eye-view grids")
    args = ap.parse_args()
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)

    sim = WarehouseSim2D(scenario=args.scenario, seed=args.seed, terminate_on_collision=False)
    ctrl = PathFollower.from_scenario(sim.world, sim.robot, sim.scenario)
    field = None if args.no_safety else ProtectiveField(sim.robot)
    total = 0
    for ep in range(args.episodes):
        s = sim.reset(seed=args.seed * 100000 + ep)
        ctrl.reset(s["pose"], s["goal"], sim.rng)
        rec = {k: [] for k in ("t", "scan", "scan_raw", "cmd", "cmd_hw", "odom_pose", "odom_twist", "gt_pose",
                               "gt_twist", "health", "payload", "peds", "bev")}
        while not sim.done:
            v, w = ctrl.act(s["pose"], s["t"])
            if field is not None:
                v, w, _ = field.filter(v, w, s["scan"], s["angles"], s["odom_vel"][0])
            s = sim.step(v, w)
            th = s["pose"][2]
            peds = np.zeros((K_PEDS, 7), np.float32)
            p = s["peds"][:K_PEDS]
            peds[: len(p), :6] = p
            peds[: len(p), 6] = 1.0
            rec["t"].append(s["t"])
            rec["scan"].append(s["scan"])
            rec["scan_raw"].append(s["scan_clean"])
            rec["cmd"].append(s["cmd"])
            rec["cmd_hw"].append(s["applied_cmd"])
            rec["odom_pose"].append(s["odom_pose"])
            rec["odom_twist"].append(s["odom_vel"])
            rec["gt_pose"].append(s["pose"])
            rec["gt_twist"].append((s["true_vel"][0] * np.cos(th), s["true_vel"][0] * np.sin(th), s["true_vel"][1]))
            rec["health"].append(s["health"])
            rec["payload"].append(s["payload"])
            rec["peds"].append(peds)
            if args.bev:
                rec["bev"].append(scan_to_bev(s["scan"], s["angles"]))
        meta = {"scenario": args.scenario, "faults": sim.health.faults, "severities": sim.health.severities,
                "onset_time": sim.health.onset_time, "density": str(sim.density), "goal": s["goal"].tolist(),
                "outcome": sim.outcome, "metrics": sim.metrics.summary(sim.outcome), "dt": sim.dt}
        arrays = {k: np.asarray(v, np.float32) for k, v in rec.items() if v}
        path = os.path.join(out, f"{args.scenario}_ep{ep:05d}.npz")
        np.savez_compressed(path, angles=sim.angles.astype(np.float32), meta=json.dumps(meta), **arrays)
        total += len(rec["t"])
        print(f"[{ep + 1}/{args.episodes}] {len(rec['t'])} steps  faults={sim.health.faults or 'none'} "
              f"onset={sim.health.onset_time:.1f}s payload={sim.payload:.0f}kg outcome={sim.outcome} -> {path}")
    print(f"total samples: {total}")


if __name__ == "__main__":
    main()
