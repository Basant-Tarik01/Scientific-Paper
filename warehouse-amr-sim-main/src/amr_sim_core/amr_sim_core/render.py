"""Matplotlib rendering of the world and of a running 2D simulation."""
from __future__ import annotations

import numpy as np

COLORS = {
    "wall": "#5b5b5b", "rack": "#1f5fa8", "pillar": "#7a7a7a", "station": "#b5651d",
    "pallet": "#c8a165", "charger": "#2e8b57", "obstacle": "#666666",
    "marking_yellow": "#e8c21a", "marking_green": "#3cb371",
}
ZONE_COLORS = {"charging": "#2e8b57", "dock": "#c8a165", "packing": "#b5651d",
               "transport_aisle": "#999999", "picking_A": "#1f5fa8", "picking_B": "#1f5fa8"}


def draw_world(world, ax, show_graph=False, show_zones=True, show_labels=False):
    import matplotlib.patches as mp
    if show_zones:
        for name, (x0, y0, x1, y1) in world.zones.items():
            ax.add_patch(mp.Rectangle((x0, y0), x1 - x0, y1 - y0, fc=ZONE_COLORS.get(name, "#cccccc"),
                                      alpha=0.07, ec="none"))
            if show_labels:
                ax.text(x0 + 0.2, y1 - 0.5, name, fontsize=7, color="#555555")
    for m in world.markings:
        x0, y0, x1, y1 = m.aabb
        ax.add_patch(mp.Rectangle((x0, y0), x1 - x0, y1 - y0, fc=COLORS.get(m.kind, "#cccccc"), ec="none", alpha=0.8))
    for b in world.boxes:
        x0, y0, x1, y1 = b.aabb
        ax.add_patch(mp.Rectangle((x0, y0), x1 - x0, y1 - y0, fc=COLORS.get(b.kind, "#666666"), ec="black", lw=0.4))
        if show_labels and b.kind not in ("wall",):
            ax.text(b.cx, b.cy, b.name.replace("rack_", ""), fontsize=5, ha="center", va="center", color="white")
    if show_graph:
        g = world.graph
        for i, nb in enumerate(g.neighbors):
            for j in nb:
                if j > i:
                    ax.plot(*g.nodes[[i, j]].T, color="#bbbbbb", lw=0.4, zorder=0)
        ax.scatter(*g.nodes.T, s=3, color="#999999", zorder=1)
    ax.set_xlim(-0.5, world.width + 0.5)
    ax.set_ylim(-0.5, world.height + 0.5)
    ax.set_aspect("equal")


def render_sim(sim, ax=None, as_array=False):
    import matplotlib
    if as_array:
        matplotlib.use("Agg", force=False)
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt
    from .geometry import rect_corners

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 7), dpi=80)
    else:
        fig = ax.figure
    draw_world(sim.world, ax)
    r = sim.robot
    x, y, th = sim.pose
    # scan
    sc = sim.scan
    ok = np.isfinite(sc)
    a = sim.angles[ok] + th
    ax.scatter(x + sc[ok] * np.cos(a), y + sc[ok] * np.sin(a), s=1.5, c="red", zorder=4)
    # robot
    ax.add_patch(mp.Polygon(rect_corners(x, y, th, r.half_length, r.half_width), fc="#f0f0f0", ec="black", zorder=5))
    ax.arrow(x, y, 0.5 * np.cos(th), 0.5 * np.sin(th), width=0.04, color="black", zorder=6)
    # pedestrians
    for p in sim.crowd.states():
        ax.add_patch(mp.Circle(p[:2], p[4], fc="#ff8c00" if p[5] else "#d62728", ec="black", lw=0.5, zorder=5))
        ax.arrow(p[0], p[1], 0.5 * p[2], 0.5 * p[3], width=0.02, color="black", zorder=6)
    ax.plot(*sim.goal, marker="*", ms=14, color="gold", mec="black", zorder=6)
    h = sim.health
    ax.set_title(f"t={sim.t:5.1f}s  v={sim.v:4.2f}  faults={','.join(h.faults) or 'none'}  "
                 f"payload={sim.payload:.0f}kg  density={sim.density}  outcome={sim.outcome}", fontsize=9)
    if as_array:
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        if own_fig:
            plt.close(fig)
        return img
    return ax
