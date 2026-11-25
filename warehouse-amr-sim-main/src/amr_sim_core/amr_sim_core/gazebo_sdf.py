"""Generate a Gazebo Classic (SDF 1.6) world from a :class:`World`.

Every collision geometry is exactly the box used by the 2D simulator and the Nav2 map, so the three
representations cannot drift apart. Visual detail (rack uprights, beams, cartons, pallets, floor
markings, worker models) is added on top and never affects collisions.
"""
from __future__ import annotations

import zlib

import numpy as np

RGBA = {
    "floor": (0.62, 0.62, 0.60, 1), "wall": (0.86, 0.86, 0.84, 1), "upright": (0.12, 0.33, 0.62, 1),
    "beam": (0.93, 0.45, 0.10, 1), "carton": (0.72, 0.55, 0.35, 1), "carton2": (0.62, 0.47, 0.30, 1),
    "pillar": (0.70, 0.70, 0.70, 1), "stripe": (0.95, 0.80, 0.05, 1), "station": (0.55, 0.38, 0.22, 1),
    "station_top": (0.80, 0.80, 0.78, 1), "pallet": (0.76, 0.60, 0.38, 1), "wrap": (0.85, 0.88, 0.92, 1),
    "charger": (0.18, 0.55, 0.34, 1), "marking_yellow": (0.95, 0.80, 0.05, 1),
    "marking_green": (0.20, 0.70, 0.40, 1), "skin": (0.87, 0.70, 0.55, 1), "trousers": (0.15, 0.17, 0.25, 1),
    "vest_orange": (1.0, 0.45, 0.0, 1), "vest_yellow": (0.85, 1.0, 0.0, 1), "helmet": (0.95, 0.95, 0.95, 1),
}


def _mat(color):
    r, g, b, a = RGBA[color] if isinstance(color, str) else color
    return (f"<material><ambient>{r:.3f} {g:.3f} {b:.3f} {a}</ambient><diffuse>{r:.3f} {g:.3f} {b:.3f} {a}</diffuse>"
            f"<specular>0.1 0.1 0.1 1</specular></material>")


def _box_visual(name, x, y, z, sx, sy, sz, color):
    return (f"<visual name='{name}'><pose>{x:.4f} {y:.4f} {z:.4f} 0 0 0</pose><geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}"
            f"</size></box></geometry>{_mat(color)}</visual>")


def _cyl_visual(name, x, y, z, r, length, color):
    return (f"<visual name='{name}'><pose>{x:.4f} {y:.4f} {z:.4f} 0 0 0</pose><geometry><cylinder><radius>{r}</radius>"
            f"<length>{length}</length></cylinder></geometry>{_mat(color)}</visual>")


def _box_collision(name, x, y, z, sx, sy, sz):
    return (f"<collision name='{name}'><pose>{x:.4f} {y:.4f} {z:.4f} 0 0 0</pose><geometry><box><size>{sx:.4f} {sy:.4f} "
            f"{sz:.4f}</size></box></geometry></collision>")


def _static_model(name, x, y, inner):
    return (f"<model name='{name}'><static>true</static><pose>{x:.4f} {y:.4f} 0 0 0 0</pose>"
            f"<link name='link'>{inner}</link></model>")


def _rack(b):
    rng = np.random.default_rng(zlib.crc32(b.name.encode()))
    levels = int(b.meta.get("levels", 4))
    sx, sy, h = b.sx, b.sy, b.height
    out = [_box_collision("collision", 0, 0, h / 2, sx, sy, h)]
    n_bays = max(1, int(round(sx / 1.8)))
    bay = sx / n_bays
    up = 0.08
    for i in range(n_bays + 1):
        x = -sx / 2 + up / 2 + i * (sx - up) / n_bays
        for j, y in enumerate((-sy / 2 + up / 2, sy / 2 - up / 2)):
            out.append(_box_visual(f"upright_{i}_{j}", x, y, h / 2, up, up, h, "upright"))
    for lv in range(levels):
        z = 0.12 + lv * (h - 0.2) / levels
        out.append(_box_visual(f"beam_{lv}", 0, 0, z, sx, sy - 0.02, 0.05, "beam"))
        for k in range(n_bays):
            if rng.random() < 0.15:          # some empty locations
                continue
            cx = -sx / 2 + (k + 0.5) * bay
            w, d = bay * rng.uniform(0.6, 0.85), sy * rng.uniform(0.7, 0.9)
            ch = min((h - 0.2) / levels - 0.12, rng.uniform(0.35, 0.5))
            out.append(_box_visual(f"carton_{lv}_{k}", cx, 0, z + 0.025 + ch / 2, w, d, ch,
                                   "carton" if rng.random() < 0.5 else "carton2"))
    return "".join(out)


def _generic(b):
    sx, sy, h = b.sx, b.sy, b.height
    out = [_box_collision("collision", 0, 0, h / 2, sx, sy, h)]
    if b.kind == "pillar":
        out.append(_box_visual("body", 0, 0, h / 2, sx, sy, h, "pillar"))
        for k in range(4):
            out.append(_box_visual(f"stripe_{k}", 0, 0, 0.1 + 0.25 * k, sx + 0.01, sy + 0.01, 0.12,
                                   "stripe" if k % 2 == 0 else (0.05, 0.05, 0.05, 1)))
    elif b.kind == "station":
        out.append(_box_visual("base", 0, 0, (h - 0.05) / 2, sx - 0.04, sy - 0.04, h - 0.05, "station"))
        out.append(_box_visual("top", 0, 0, h - 0.025, sx, sy, 0.05, "station_top"))
        out.append(_box_visual("parcel", 0.3 * sx / 2, 0, h + 0.12, 0.35, 0.3, 0.24, "carton"))
    elif b.kind == "pallet":
        out.append(_box_visual("pallet", 0, 0, 0.07, sx, sy, 0.14, "pallet"))
        out.append(_box_visual("load", 0, 0, 0.14 + (h - 0.14) / 2, sx - 0.06, sy - 0.06, h - 0.14, "wrap"))
    elif b.kind == "charger":
        out.append(_box_visual("body", 0, 0, h / 2, sx, sy, h, "charger"))
    else:
        out.append(_box_visual("body", 0, 0, h / 2, sx, sy, h, "wall" if b.kind == "wall" else "pillar"))
    return "".join(out)


def pedestrian_model(i: int, park_xy) -> str:
    vest = "vest_orange" if i % 2 == 0 else "vest_yellow"
    v = [
        _cyl_visual("leg_l", 0, 0.09, 0.43, 0.075, 0.86, "trousers"),
        _cyl_visual("leg_r", 0, -0.09, 0.43, 0.075, 0.86, "trousers"),
        _box_visual("torso", 0, 0, 1.17, 0.24, 0.42, 0.62, vest),
        _cyl_visual("arm_l", 0, 0.25, 1.15, 0.05, 0.6, vest),
        _cyl_visual("arm_r", 0, -0.25, 1.15, 0.05, 0.6, vest),
        f"<visual name='head'><pose>0 0 1.62 0 0 0</pose><geometry><sphere><radius>0.11</radius></sphere></geometry>{_mat('skin')}</visual>",
        f"<visual name='helmet'><pose>0 0 1.69 0 0 0</pose><geometry><sphere><radius>0.12</radius></sphere></geometry>{_mat('helmet')}</visual>",
    ]
    return (f"<model name='ped_{i}'><static>false</static><pose>{park_xy[0]:.2f} {park_xy[1]:.2f} 0 0 0 0</pose>"
            f"<link name='link'><kinematic>true</kinematic><gravity>false</gravity>"
            f"<inertial><mass>70</mass><inertia><ixx>15</ixx><iyy>15</iyy><izz>1</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>"
            f"<collision name='collision'><pose>0 0 0.875 0 0 0</pose><geometry><cylinder><radius>0.25</radius><length>1.75</length></cylinder></geometry></collision>"
            f"{''.join(v)}</link></model>")


def park_position(i: int):
    """Where inactive pedestrians wait (outside the building, beyond LiDAR range)."""
    return (-20.0 - 1.5 * i, -20.0)


def world_to_sdf(world) -> str:
    parts = []
    W, H = world.width, world.height
    # floor (slightly larger than the building) and floor markings
    floor = (f"<model name='warehouse_floor'><static>true</static><link name='link'>"
             f"<collision name='collision'><pose>{W / 2} {H / 2} -0.05 0 0 0</pose><geometry><box><size>{W + 4} {H + 4} 0.1</size></box></geometry>"
             f"<surface><friction><ode><mu>0.9</mu><mu2>0.9</mu2></ode></friction></surface></collision>"
             f"{_box_visual('visual', W / 2, H / 2, -0.05, W + 4, H + 4, 0.1, 'floor')}")
    floor += "".join(_box_visual(m.name, m.cx, m.cy, 0.001, m.sx, m.sy, 0.002, m.kind) for m in world.markings)
    floor += "</link></model>"
    parts.append(floor)
    for b in world.boxes:
        inner = _rack(b) if b.kind == "rack" else _generic(b)
        parts.append(_static_model(b.name, b.cx, b.cy, inner))
    n_peds = int(world.ped_spec.get("max", 16))
    parts += [pedestrian_model(i, park_position(i)) for i in range(n_peds)]
    return f"""<?xml version="1.0"?>
<!-- GENERATED by amr_sim_core.tools.generate_assets from {world.name}.yaml - do not edit by hand. -->
<sdf version="1.6">
  <world name="{world.name}">
    <physics type="ode">
      <max_step_size>0.002</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>500</real_time_update_rate>
    </physics>
    <gravity>0 0 -9.81</gravity>
    <scene>
      <ambient>0.45 0.45 0.45 1</ambient>
      <background>0.7 0.75 0.8 1</background>
      <shadows>false</shadows>
    </scene>
    <light name="sun" type="directional">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.6 0.6 0.6 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <direction>-0.3 0.2 -1.0</direction>
    </light>
    <gui fullscreen="0">
      <camera name="user_camera">
        <pose>{W / 2:.1f} {-8:.1f} 22 0 0.95 1.5708</pose>
      </camera>
    </gui>
    <!-- /gazebo/set_entity_state, /gazebo/get_entity_state (used to teleport the robot between episodes) -->
    <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
      <ros><namespace>/gazebo</namespace></ros>
      <update_rate>1.0</update_rate>
    </plugin>
    <!-- moves the ped_<i> models to the states published on /pedestrians by pedestrian_sim -->
    <plugin name="pedestrian_pose" filename="libamr_pedestrian_pose_plugin.so">
      <model_prefix>ped_</model_prefix>
      <park_x>-20.0</park_x>
      <park_y>-20.0</park_y>
      <park_spacing>1.5</park_spacing>
    </plugin>
    {chr(10).join('    ' + p for p in parts)}
  </world>
</sdf>
"""
