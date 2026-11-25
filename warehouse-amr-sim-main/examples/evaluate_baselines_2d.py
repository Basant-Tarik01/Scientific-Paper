#!/usr/bin/env python3
"""Evaluate classical baselines in the fast 2D simulator over a fault-type x severity x density grid.

Controller: graph route + pure pursuit (not human-aware), combined with a safety layer:
  none              - no safety layer
  protective_field  - fixed industrial protective/warning field (A7 baseline)
  ttc               - reactive time-to-collision rule (A7 baseline)

    python3 examples/evaluate_baselines_2d.py --episodes 20 --out results/baselines_2d.csv

Prints a summary table (collision rate, near-miss events per episode, success rate, mean time) per
safety layer and fault severity bin. The same metric definitions are used in Gazebo (metrics_monitor).
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "amr_sim_core"))
from amr_sim_core.controllers import PathFollower  # noqa: E402
from amr_sim_core.faults import HealthState, apply_fault  # noqa: E402
from amr_sim_core.safety import ProtectiveField, TTCRule  # noqa: E402
from amr_sim_core.sim2d import WarehouseSim2D  # noqa: E402


def run_episode(sim, ctrl, layer, options, seed):
    s = sim.reset(seed=seed, options=options)
    ctrl.reset(s["pose"], s["goal"], sim.rng)
    while not sim.done:
        v, w = ctrl.act(s["pose"], s["t"])
        if layer is not None:
            v, w, _ = layer.filter(v, w, s["scan"], s["angles"], s["odom_vel"][0])
        s = sim.step(v, w)
    return sim.metrics.summary(sim.outcome)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=10, help="episodes per grid cell")
    ap.add_argument("--faults", nargs="+", default=["none", "wheel_gain", "cmd_latency", "lidar_dropout", "torque_loss"])
    ap.add_argument("--severities", nargs="+", type=float, default=[0.3, 0.6, 0.9])
    ap.add_argument("--densities", nargs="+", default=["medium", "high"])
    ap.add_argument("--layers", nargs="+", default=["none", "protective_field", "ttc"])
    ap.add_argument("--payload", type=float, default=30.0)
    ap.add_argument("--out", default="results/baselines_2d.csv")
    args = ap.parse_args()

    sim = WarehouseSim2D(scenario="a1_eval_heldout", seed=0)
    ctrl = PathFollower(sim.world, sim.robot, speed_scale=(0.8, 0.8))
    layers = {"none": None, "protective_field": ProtectiveField(sim.robot), "ttc": TTCRule(sim.robot)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows, agg = [], defaultdict(list)
    for layer_name in args.layers:
        for fault in args.faults:
            for sev in ([0.0] if fault == "none" else args.severities):
                for density in args.densities:
                    for ep in range(args.episodes):
                        seed = 1000 * ep + 7               # same start/goal/crowd across cells
                        rng = np.random.default_rng(seed)
                        h = HealthState() if fault == "none" else apply_fault(HealthState(), fault, sev, rng)
                        m = run_episode(sim, ctrl, layers[layer_name],
                                        {"health": h, "payload": args.payload, "density": density}, seed)
                        row = {"layer": layer_name, "fault": fault, "severity": sev, "density": density, "episode": ep, **m}
                        rows.append(row)
                        agg[(layer_name, fault, sev)].append(m)
                print(f"done: layer={layer_name:16s} fault={fault}")
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{'layer':16s} {'fault':14s} {'sev':>4s} {'collision':>9s} {'near-miss/ep':>12s} {'success':>8s} {'time[s]':>8s}")
    for (layer_name, fault, sev), ms in agg.items():
        coll = np.mean([m["outcome"].startswith("collision") for m in ms])
        nm = np.mean([m["near_miss_events"] for m in ms])
        succ = np.mean([m["success"] for m in ms])
        t = np.mean([m["time"] for m in ms if m["success"]]) if succ > 0 else float("nan")
        print(f"{layer_name:16s} {fault:14s} {sev:4.1f} {coll:9.2f} {nm:12.2f} {succ:8.2f} {t:8.1f}")
    print(f"\nper-episode results: {args.out}")


if __name__ == "__main__":
    main()
