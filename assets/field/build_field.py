"""Build the full-size field artwork and the textured 3D model."""
from pathlib import Path
from hashlib import sha256
import json

from PIL import Image, ImageDraw, ImageFont
import trimesh


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def build(config, label_font):
    width, height = config["texture_pixels"]
    image = Image.new("RGB", (width, height), config["background_hex"])
    draw = ImageDraw.Draw(image)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}mm" height="{height}mm" '
           f'viewBox="0 0 {width} {height}">',
           f'<rect width="{width}" height="{height}" fill="{config["background_hex"]}"/>']
    styles = config["line_widths_mm"]
    on, off = config["dash_pattern_mm"]
    for region in config["regions"]:
        x, y, w, h = region["xywh_mm"]
        top = height-y-h
        fill = region.get("fill_hex")
        stroke = region.get("stroke_hex")
        name = region["name"].split("_")[0]
        if fill:
            draw.rectangle((x, top, x+w-1, top+h-1), fill=fill)
        if region.get("dashed"):
            for start, step, length in (((x, top), (1, 0), w), ((x, top+h), (1, 0), w),
                                         ((x, top), (0, 1), h), ((x+w, top), (0, 1), h)):
                for d in range(0, length, on+off):
                    end = min(d+on, length)
                    draw.line(tuple(tuple(a+k*b for a,b in zip(start, step)) for k in (d, end)),
                              fill=stroke, width=styles["zone_dashed"])
            attrs = f'stroke="{stroke}" stroke-width="{styles["zone_dashed"]}" stroke-dasharray="{on} {off}"'
        elif stroke:
            pen = styles[f"{name}_outline"]
            draw.line([(x, top), (x+w, top), (x+w, top+h), (x, top+h), (x, top)],
                      fill=stroke, width=pen)
            attrs = f'stroke="{stroke}" stroke-width="{pen}"'
        else:
            attrs = ""
        svg.append(f'<rect id="{region["name"]}" x="{x}" y="{top}" width="{w}" height="{h}" '
                   f'fill="{fill or "none"}" {attrs}/>')
    pen = styles["outer_border"]
    draw.rectangle((0, 0, width-1, height-1), outline=config["outer_border_hex"], width=pen)
    svg.append(f'<rect x="{pen/2}" y="{pen/2}" width="{width-pen}" height="{height-pen}" fill="none" '
               f'stroke="{config["outer_border_hex"]}" stroke-width="{pen}"/>')
    svg.append("</svg>")
    (HERE/"field.svg").write_text("\n".join(svg)+"\n", encoding="utf-8")
    image.save(HERE/"field.png", optimize=True)
    build_glb(config, image)
    build_preview(config, image, label_font)


def build_glb(config, image):
    # OBJ is the shared visual floor; metres with Z up and UV coordinates.
    floor = trimesh.load_mesh(HERE/"field.obj", process=False)
    x, y = config["field_size_m"]
    floor.visual = trimesh.visual.TextureVisuals(
        uv=floor.vertices[:, :2]/[x, y],
        material=trimesh.visual.material.PBRMaterial(baseColorTexture=image, roughnessFactor=1.))
    scene = trimesh.Scene()
    scene.add_geometry(floor, node_name="floor")
    h, t = config["wall_height_mm"]*.001, config["wall_thickness_mm"]*.001
    color = tuple(bytes.fromhex(config["wall_hex"].lstrip("#")))+(255,)
    for name, extents, center in (
        ("wall_west", (t, y+2*t, h), (-t/2, y/2, h/2)),
        ("wall_east", (t, y+2*t, h), (x+t/2, y/2, h/2)),
        ("wall_south", (x, t, h), (x/2, -t/2, h/2)),
        ("wall_north", (x, t, h), (x/2, y+t/2, h/2)),
    ):
        box = trimesh.creation.box(extents=extents)
        box.apply_translation(center)
        box.visual.face_colors = color
        scene.add_geometry(box, node_name=name)
    scene.export(HERE/"field.glb")


def build_preview(config, image, label_font):
    scale = .50
    ox, oy = 135, 190
    canvas = Image.new("RGB", (1770, 1330), "#eaf0f4")
    canvas.paste(image.resize((1500, 1000), Image.Resampling.LANCZOS), (ox, oy))
    draw = ImageDraw.Draw(canvas)
    font = lambda size: ImageFont.truetype(str(label_font), size)
    ink = "#263949"
    draw.text((ox, 48), "智能车光电象棋对抗 · 场地模型", font=font(48), fill=ink)
    draw.text((ox, 109), "区赛参考细则  /  单位 mm  /  挡板高 300，所有区与赛面齐平", font=font(25), fill=ink)
    for coord, label in (
        ((200, 1000), "甲队球门\n400 × 600"), ((2800, 1000), "乙队球门\n400 × 600"),
        ((200, 200), "甲队发车区\n300 × 300"), ((2800, 1800), "乙队发车区\n300 × 300"),
        ((1500, 1000), "中央摆棋区\n600 × 600"),
    ):
        draw.multiline_text((ox+coord[0]*scale, oy+(2000-coord[1])*scale), label, font=font(24),
                            fill=ink, anchor="mm", align="center", spacing=8)
    draw.text((885, 151), "←                          内净长 3000                          →",
              font=font(24), fill=ink, anchor="mm")
    tag = Image.new("RGBA", (300, 50))
    ImageDraw.Draw(tag).text((150, 24), "内净宽 2000", font=font(24), fill=ink, anchor="mm")
    tag = tag.rotate(90, expand=True)
    canvas.paste(tag, (80, 540), tag)
    draw.text((ox, 1242), "发车区靠墙留边按图示取 50 mm，实际放样以赛区确认尺寸为准。",
              font=font(24), fill=ink)
    canvas.save(HERE/"preview.png", optimize=True)


def record(config):
    config["files"] = {}
    paths = [HERE/n for n in ("field.obj", "field.png", "field.svg", "field.glb", "preview.png")]
    for path in paths:
        config["files"][path.name] = dict(bytes=path.stat().st_size, sha256=sha256(path.read_bytes()).hexdigest())
    file = HERE/"field.json"
    file.write_text(json.dumps(config, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    path = REPO/"assets/manifests/models.json"
    manifest = json.loads(path.read_text())
    manifest["source_selection"]["field"] = "2026-11-17 competition notice, figure 1"
    keep = [entry for entry in manifest["files"] if not entry["path"].startswith("assets/field/")]
    manifest["files"] = sorted(keep + [dict(path=str(p.relative_to(REPO)), bytes=p.stat().st_size,
        sha256=sha256(p.read_bytes()).hexdigest()) for p in [file, *paths]], key=lambda item: item["path"])
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-font", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((HERE/"field.json").read_text())
    build(config, args.label_font)
    record(config)
