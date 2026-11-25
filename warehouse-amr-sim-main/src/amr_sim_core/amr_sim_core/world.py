"""Warehouse world model: loads the YAML layout, expands it into boxes, builds the waypoint graph,
rasterises occupancy maps and samples start/goal poses."""
from __future__ import annotations

import heapq
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from .geometry import point_aabb_distance, segment_clearance

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def resolve_data_file(name_or_path: str, kind: str) -> str:
    """Return a path for a world/scenario/robot given either a file path or a bundled name."""
    if os.path.isfile(name_or_path):
        return name_or_path
    candidate = os.path.join(DATA_DIR, kind, name_or_path)
    if not candidate.endswith(".yaml"):
        candidate += ".yaml"
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(f"{kind[:-1]} '{name_or_path}' not found (looked in {os.path.dirname(candidate)})")


@dataclass
class Box:
    name: str
    kind: str              # wall | rack | pillar | station | pallet | charger
    cx: float
    cy: float
    sx: float
    sy: float
    height: float
    collision: bool = True
    meta: dict = field(default_factory=dict)

    @property
    def aabb(self) -> Tuple[float, float, float, float]:
        return (self.cx - self.sx / 2, self.cy - self.sy / 2, self.cx + self.sx / 2, self.cy + self.sy / 2)


@dataclass
class WaypointGraph:
    nodes: np.ndarray                  # (K, 2)
    clearance: np.ndarray              # (K,)
    neighbors: List[List[int]]

    def nearest(self, x: float, y: float, boxes: Optional[np.ndarray] = None) -> int:
        d = np.hypot(self.nodes[:, 0] - x, self.nodes[:, 1] - y)
        if boxes is None:
            return int(np.argmin(d))
        for idx in np.argsort(d)[:12]:   # prefer a node that is directly reachable
            if segment_clearance((x, y), self.nodes[idx], boxes) > 0.25:
                return int(idx)
        return int(np.argmin(d))

    def shortest_path(self, start: int, goal: int) -> List[int]:
        """Dijkstra on Euclidean edge lengths. Returns node indices (empty if unreachable)."""
        dist = {start: 0.0}
        prev: Dict[int, int] = {}
        heap = [(0.0, start)]
        while heap:
            d, u = heapq.heappop(heap)
            if u == goal:
                break
            if d > dist.get(u, np.inf):
                continue
            for v in self.neighbors[u]:
                nd = d + float(np.hypot(*(self.nodes[u] - self.nodes[v])))
                if nd < dist.get(v, np.inf):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
        if goal not in dist:
            return []
        path = [goal]
        while path[-1] != start:
            path.append(prev[path[-1]])
        return path[::-1]

    def path_length(self, path: Sequence[int]) -> float:
        if len(path) < 2:
            return 0.0
        pts = self.nodes[list(path)]
        return float(np.sum(np.hypot(*np.diff(pts, axis=0).T)))


class World:
    """Expanded warehouse layout."""

    def __init__(self, spec: dict, source: str = ""):
        self.spec = spec
        self.source = source
        self.name: str = spec["name"]
        self.description: str = spec.get("description", "")
        self.width, self.height = map(float, spec["size"])
        self.boxes: List[Box] = []
        self.markings: List[Box] = []      # floor markings (visual only)
        self.zones: Dict[str, Tuple[float, float, float, float]] = {
            k: tuple(map(float, v)) for k, v in spec.get("zones", {}).items()
        }
        self.ped_spec: dict = spec.get("pedestrians", {"max": 16, "densities": {"low": 4}})
        self.robot_spawn: dict = spec.get("robot_spawn", {"x": 1.0, "y": 1.0, "yaw": 0.0})
        self._build_walls()
        self._build_racks()
        self._build_obstacles()
        self.aabbs = np.array([b.aabb for b in self.boxes if b.collision], dtype=float)
        gspec = spec.get("graph", {})
        self.graph = self._build_graph(float(gspec.get("spacing", 1.0)), float(gspec.get("clearance", 0.6)))

    # ------------------------------------------------------------------ construction
    def _build_walls(self):
        w = self.spec.get("walls", {})
        t = float(w.get("thickness", 0.2))
        h = float(w.get("height", 3.0))
        W, H = self.width, self.height
        self.boxes += [
            Box("wall_south", "wall", W / 2, -t / 2, W + 2 * t, t, h),
            Box("wall_north", "wall", W / 2, H + t / 2, W + 2 * t, t, h),
            Box("wall_west", "wall", -t / 2, H / 2, t, H, h),
            Box("wall_east", "wall", W + t / 2, H / 2, t, H, h),
        ]

    def _build_racks(self):
        for block in self.spec.get("rack_blocks", []):
            ox, oy = map(float, block["origin"])
            L = float(block["row_length"])
            D = float(block["row_depth"])
            A = float(block["aisle_width"])
            h = float(block.get("height", 2.4))
            levels = int(block.get("levels", 4))
            gaps = sorted(((float(g["at"]) - float(g["width"]) / 2, float(g["at"]) + float(g["width"]) / 2)
                           for g in block.get("gaps", [])))
            for r in range(int(block["rows"])):
                y0 = oy + r * (D + A)
                # split the row at the gaps
                segments, start = [], 0.0
                for g0, g1 in gaps:
                    segments.append((start, g0))
                    start = g1
                segments.append((start, L))
                for s, (a, b) in enumerate(segments):
                    if b - a < 0.2:
                        continue
                    name = f"rack_{block['name']}{r + 1}_{s + 1}"
                    self.boxes.append(Box(name, "rack", ox + (a + b) / 2, y0 + D / 2, b - a, D, h,
                                          meta={"levels": levels, "block": block["name"], "row": r}))
                    # yellow floor lines 0.15 m in front of both long faces of the rack
                    for side, yl in (("s", y0 - 0.15), ("n", y0 + D + 0.15)):
                        self.markings.append(Box(f"mark_{name}_{side}", "marking_yellow",
                                                 ox + (a + b) / 2, yl, b - a, 0.06, 0.002, collision=False))

    def _build_obstacles(self):
        for ob in self.spec.get("obstacles", []):
            cx, cy = map(float, ob["center"])
            sx, sy = map(float, ob["size"])
            self.boxes.append(Box(ob["name"], ob.get("kind", "obstacle"), cx, cy, sx, sy,
                                  float(ob.get("height", 1.0)), bool(ob.get("collision", True))))
        for name, (x0, y0, x1, y1) in self.zones.items():
            if name in ("dock", "charging"):
                for i, (cx, cy, sx, sy) in enumerate((((x0 + x1) / 2, y0, x1 - x0, 0.08),
                                                       ((x0 + x1) / 2, y1, x1 - x0, 0.08),
                                                       (x0, (y0 + y1) / 2, 0.08, y1 - y0),
                                                       (x1, (y0 + y1) / 2, 0.08, y1 - y0))):
                    self.markings.append(Box(f"mark_{name}_{i}", "marking_green", cx, cy, sx, sy, 0.002, False))

    def _build_graph(self, spacing: float, clearance: float) -> WaypointGraph:
        xs = np.arange(spacing / 2, self.width, spacing)
        ys = np.arange(spacing / 2, self.height, spacing)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        clear = point_aabb_distance(pts[:, 0], pts[:, 1], self.aabbs).min(axis=1)
        keep = clear >= clearance
        nodes, clear = pts[keep], clear[keep]
        index = {(round(p[0], 3), round(p[1], 3)): i for i, p in enumerate(nodes)}
        neighbors: List[List[int]] = [[] for _ in range(len(nodes))]
        for i, p in enumerate(nodes):
            for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
                key = (round(p[0] + dx * spacing, 3), round(p[1] + dy * spacing, 3))
                j = index.get(key)
                if j is None:
                    continue
                if segment_clearance(p, nodes[j], self.aabbs) >= clearance * 0.9:
                    neighbors[i].append(j)
                    neighbors[j].append(i)
        return WaypointGraph(nodes, clear, neighbors)

    # ------------------------------------------------------------------ queries
    def clearance(self, x, y) -> np.ndarray:
        return point_aabb_distance(x, y, self.aabbs).min(axis=1)

    def is_free(self, x: float, y: float, margin: float = 0.0) -> bool:
        return bool(self.clearance(x, y)[0] > margin)

    def in_zone(self, zone: str, x: float, y: float) -> bool:
        x0, y0, x1, y1 = self.zones[zone]
        return x0 <= x <= x1 and y0 <= y <= y1

    def nodes_in_zone(self, zone: str) -> np.ndarray:
        x0, y0, x1, y1 = self.zones[zone]
        n = self.graph.nodes
        return np.where((n[:, 0] >= x0) & (n[:, 0] <= x1) & (n[:, 1] >= y0) & (n[:, 1] <= y1))[0]

    def sample_start_goal(self, rng: np.random.Generator, min_dist: float = 5.0, max_dist: float = 25.0,
                          min_clearance: float = 0.7, max_tries: int = 200):
        """Sample start pose (x, y, yaw) and goal (x, y) on the graph, connected and within distance bounds."""
        cand = np.where(self.graph.clearance >= min_clearance)[0]
        for _ in range(max_tries):
            i, j = rng.choice(cand, 2, replace=False)
            path = self.graph.shortest_path(int(i), int(j))
            if not path:
                continue
            length = self.graph.path_length(path)
            if min_dist <= length <= max_dist:
                s, g = self.graph.nodes[i], self.graph.nodes[j]
                # face roughly along the first path segment, with noise
                nxt = self.graph.nodes[path[min(2, len(path) - 1)]]
                yaw = float(np.arctan2(nxt[1] - s[1], nxt[0] - s[0]) + rng.normal(0.0, 0.5))
                return (float(s[0]), float(s[1]), yaw), (float(g[0]), float(g[1])), path
        raise RuntimeError("could not sample a start/goal pair; relax the distance bounds")

    # ------------------------------------------------------------------ maps
    def rasterize(self, resolution: float = 0.05, margin: float = 1.0):
        """Occupancy grid (row 0 = top, like a PGM image): 1 = occupied. Returns (grid, origin_xy)."""
        x0, y0 = -margin, -margin
        nx = int(round((self.width + 2 * margin) / resolution))
        ny = int(round((self.height + 2 * margin) / resolution))
        grid = np.zeros((ny, nx), dtype=np.uint8)
        for b in self.boxes:
            if not b.collision:
                continue
            ax0, ay0, ax1, ay1 = b.aabb
            c0 = max(0, int(np.floor((ax0 - x0) / resolution)))
            c1 = min(nx, int(np.ceil((ax1 - x0) / resolution)))
            r0 = max(0, int(np.floor((ay0 - y0) / resolution)))
            r1 = min(ny, int(np.ceil((ay1 - y0) / resolution)))
            grid[r0:r1, c0:c1] = 1
        # everything outside the walls is unknown-but-occupied for planning purposes
        return grid[::-1].copy(), (x0, y0)

    def write_nav2_map(self, out_dir: str, resolution: float = 0.05, margin: float = 1.0) -> str:
        """Write <name>.pgm and <name>.yaml (Nav2 map_server format). Returns the yaml path."""
        os.makedirs(out_dir, exist_ok=True)
        grid, (ox, oy) = self.rasterize(resolution, margin)
        img = np.where(grid == 1, 0, 254).astype(np.uint8)
        pgm = os.path.join(out_dir, f"{self.name}.pgm")
        with open(pgm, "wb") as f:
            f.write(f"P5\n# generated from {self.name} by amr_sim_core\n{img.shape[1]} {img.shape[0]}\n255\n".encode())
            f.write(img.tobytes())
        yml = os.path.join(out_dir, f"{self.name}.yaml")
        with open(yml, "w") as f:
            f.write(f"image: {self.name}.pgm\nmode: trinary\nresolution: {resolution}\n"
                    f"origin: [{ox}, {oy}, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n")
        return yml


def load_world(name_or_path: str = "warehouse_small") -> World:
    path = resolve_data_file(name_or_path, "worlds")
    with open(path) as f:
        spec = yaml.safe_load(f)
    return World(spec, source=path)


def load_scenario(name_or_path: str = "nominal") -> dict:
    path = resolve_data_file(name_or_path, "scenarios")
    with open(path) as f:
        sc = yaml.safe_load(f)
    sc.setdefault("episode", {})
    sc["episode"].setdefault("max_time", 60.0)
    sc["episode"].setdefault("goal_tolerance", 0.3)
    sc["episode"].setdefault("min_goal_distance", 5.0)
    sc["episode"].setdefault("max_goal_distance", 25.0)
    sc.setdefault("pedestrians", {"density": ["medium"]})
    sc.setdefault("payload", [0.0, 0.0])
    sc.setdefault("faults", {"p_nominal": 1.0, "types": []})
    sc.setdefault("metrics", {"near_miss_distance": 0.5})
    sc["_source"] = path
    return sc
