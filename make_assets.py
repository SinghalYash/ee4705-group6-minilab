"""Generate the temporary Task 4 test scene (Student C sandbox only).

This is NOT the official Task 2 scene. It exists so Task 4 perception and
the approach controller can be developed before Task 2 is finished. Once
Student A delivers the real scene, point config/objects.yaml at it instead.

Outputs (next to this file):
  assets/stop_sign.obj, assets/stop_sign.png  - textured octagonal stop sign
  assets/ball.png                              - sports-ball texture
  test_scene.xml                               - terrain-only MJCF map
  ../config/objects.yaml                       - object positions (for d logging only)

Every object is a static body without joints, so the file is accepted by
runtime_control.MapSpec / compose_scene (strict terrain-only check).
"""
from pathlib import Path
import math

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
CONFIG = HERE.parent / "config" / "objects.yaml"

# name, class, color, x, y, yaw_deg
# (A textured sphere was tried as a "sports ball": COCO YOLO did not detect it
#  reliably at any size/texture tested, so the sandbox uses a third chair.)
# Chairs face the origin-ish so the robot sees a front/three-quarter view.
OBJECTS = [
    ("chair_green", "chair", "green", 3.0, 1.2, 200.0),
    ("chair_red", "chair", "red", 3.0, -1.2, 160.0),
    ("chair_blue", "chair", "blue", -2.5, 1.5, -30.0),
    ("stop_sign", "stop sign", "red", -1.5, -3.0, 60.0),
]

COLOR_RGBA = {
    "green": "0.10 0.60 0.15 1",
    "red": "0.75 0.08 0.08 1",
    "blue": "0.10 0.25 0.80 1",
    "yellow": "0.90 0.80 0.10 1",
}


def make_stop_sign(obj_path, png_path, radius=0.22, thickness=0.02):
    """Extruded octagon; front face UV-mapped onto a STOP texture."""
    size = 512
    img = Image.new("RGB", (size, size), (190, 20, 25))
    draw = ImageDraw.Draw(img)
    c = size / 2

    def octagon(r):
        return [
            (c + r * math.cos(math.radians(22.5 + 45 * k)),
             c + r * math.sin(math.radians(22.5 + 45 * k)))
            for k in range(8)
        ]

    draw.polygon(octagon(size * 0.50), fill=(255, 255, 255))
    draw.polygon(octagon(size * 0.46), fill=(190, 20, 25))
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(size * 0.26)
    )
    box = draw.textbbox((0, 0), "STOP", font=font)
    draw.text(
        (c - (box[2] - box[0]) / 2 - box[0], c - (box[3] - box[1]) / 2 - box[1]),
        "STOP", fill=(255, 255, 255), font=font,
    )
    img.save(png_path)

    # Sign plane is the local Y-Z plane, front face points +X.
    angles = [math.radians(22.5 + 45 * k) for k in range(8)]
    ring = [(math.cos(a), math.sin(a)) for a in angles]
    lines = ["# stop sign octagon prism"]
    verts, uvs = [], []
    for x in (thickness / 2, -thickness / 2):  # front then back
        for (cy, cz) in ring:
            verts.append((x, radius * cy, radius * cz))
    verts.append((thickness / 2, 0.0, 0.0))    # front centre (16)
    verts.append((-thickness / 2, 0.0, 0.0))   # back centre (17)
    for v in verts:
        lines.append("v %.5f %.5f %.5f" % v)
    # UVs: viewer looking at +X face along -X sees +Y on the LEFT.
    for (cy, cz) in ring:
        uvs.append((0.5 - 0.5 * cy, 0.5 + 0.5 * cz))
    uvs.append((0.5, 0.5))
    uvs.append((0.02, 0.02))  # plain red for back/sides
    for t in uvs:
        lines.append("vt %.5f %.5f" % t)
    faces = []
    for k in range(8):  # front fan
        a, b = k, (k + 1) % 8
        faces.append(((16, 8), (a, a), (b, b)))
    for k in range(8):  # back fan
        a, b = 8 + k, 8 + (k + 1) % 8
        faces.append(((17, 9), (b, 9), (a, 9)))
    for k in range(8):  # sides
        a, b = k, (k + 1) % 8
        faces.append(((a, 9), (8 + a, 9), (8 + b, 9)))
        faces.append(((a, 9), (8 + b, 9), (b, 9)))
    for f in faces:
        lines.append("f " + " ".join(f"{v + 1}/{t + 1}" for v, t in f))
    obj_path.write_text("\n".join(lines) + "\n")


def make_ball_texture(png_path):
    """Equirect-ish panel pattern so the sphere reads as a sports ball."""
    w, h = 512, 256
    img = Image.new("RGB", (w, h), (30, 70, 200))
    d = ImageDraw.Draw(img)
    for k in range(8):
        x = k * w / 8
        d.line([(x, 0), (x, h)], fill=(240, 240, 240), width=6)
    for y in (h * 0.33, h * 0.66):
        d.line([(0, y), (w, y)], fill=(240, 240, 240), width=6)
    img.save(png_path)


def quat_z(yaw_deg):
    a = math.radians(yaw_deg) / 2
    return f"{math.cos(a):.6f} 0 0 {math.sin(a):.6f}"


# Chairs are scaled down from adult size (0.9 m tall). The camera sits 0.28 m
# ahead of the trunk at ~0.49 m height, so at the "found" distance (centre
# <= 0.8 m from the trunk) a full-size chair overflows the frame and YOLO
# stops detecting it. At 0.7x the whole chair stays in view.
CHAIR_SCALE = 0.7


def chair_xml(name, x, y, yaw, color, k=CHAIR_SCALE):
    rgba = COLOR_RGBA[color]
    leg = "0.25 0.18 0.12 1"
    s = 0.22 * k  # seat half-width

    def box(sx, sy, sz, px, py, pz, c):
        return (f'<geom type="box" size="{sx*k:.4f} {sy*k:.4f} {sz*k:.4f}" '
                f'pos="{px:.4f} {py:.4f} {pz*k:.4f}" rgba="{c}"/>')

    parts = [box(0.22, 0.22, 0.025, 0, 0, 0.45, rgba)]
    for lx in (-s + 0.03 * k, s - 0.03 * k):
        for ly in (-s + 0.03 * k, s - 0.03 * k):
            parts.append(box(0.022, 0.022, 0.215, lx, ly, 0.215, leg))
    # backrest on local -X side (chair faces +X)
    bx = -s + 0.03 * k
    parts.append(box(0.022, 0.022, 0.22, bx, -s + 0.03 * k, 0.69, leg))
    parts.append(box(0.022, 0.022, 0.22, bx, s - 0.03 * k, 0.69, leg))
    parts.append(box(0.02, 0.22, 0.10, bx, 0, 0.78, rgba))
    parts.append(box(0.02, 0.19, 0.03, bx, 0, 0.60, rgba))
    inner = "\n      ".join(parts)
    return (
        f'    <body name="{name}" pos="{x} {y} 0" quat="{quat_z(yaw)}">\n'
        f"      {inner}\n    </body>"
    )


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    make_stop_sign(ASSETS / "stop_sign.obj", ASSETS / "stop_sign.png")
    make_ball_texture(ASSETS / "ball.png")

    bodies = []
    for name, cls, color, x, y, yaw in OBJECTS:
        if cls == "chair":
            bodies.append(chair_xml(name, x, y, yaw, color))
        elif cls == "sports ball":
            bodies.append(
                f'    <body name="{name}" pos="{x} {y} 0.12">\n'
                f'      <geom type="sphere" size="0.12" material="ball_mat"/>\n'
                f"    </body>"
            )
        elif cls == "stop sign":
            bodies.append(
                f'    <body name="{name}" pos="{x} {y} 0" quat="{quat_z(yaw)}">\n'
                f'      <geom type="cylinder" size="0.02 0.20" pos="-0.03 0 0.20" rgba="0.6 0.6 0.6 1"/>\n'
                f'      <geom type="mesh" mesh="stop_sign" material="stop_mat" pos="0 0 0.55"/>\n'
                f"    </body>"
            )

    xml = f"""<mujoco model="task4_test_scene">
  <!-- Student C sandbox scene. Terrain-only (no joints) so MapSpec accepts it. -->
  <compiler meshdir="assets" texturedir="assets"/>
  <asset>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.72 0.72 0.70" rgb2="0.62 0.62 0.60" width="512" height="512"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="8 8" reflectance="0.05"/>
    <texture name="stop_tex" type="2d" file="stop_sign.png"/>
    <material name="stop_mat" texture="stop_tex"/>
    <texture name="ball_tex" type="2d" file="ball.png"/>
    <material name="ball_mat" texture="ball_tex"/>
    <mesh name="stop_sign" file="stop_sign.obj"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1" material="floor_mat"/>
{chr(10).join(bodies)}
  </worldbody>
</mujoco>
"""
    (HERE / "test_scene.xml").write_text(xml)

    cfg = {
        "scene_file": "sandbox_scene/test_scene.xml",
        "note": "Ground truth for d logging/evaluation ONLY. Never used for steering.",
        "objects": [
            {"name": n, "class": c, "color": col, "x": x, "y": y}
            for n, c, col, x, y, _ in OBJECTS
        ],
    }
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("wrote", HERE / "test_scene.xml", "and", CONFIG)


if __name__ == "__main__":
    main()
