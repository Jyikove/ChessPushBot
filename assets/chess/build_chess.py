"""Export the competition chess pieces as CAD, colored GLB and MuJoCo meshes."""
from pathlib import Path
import argparse
import hashlib
import json

from contourpy import contour_generator
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates
from scipy.optimize import least_squares
from shapely import Polygon, get_parts
import trimesh
from PIL import Image, ImageDraw, ImageFont
from OCP.gp import gp_Pnt,gp_Vec,gp_Trsf,gp_Ax2,gp_Dir
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace, BRepBuilderAPI_Transform
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder,BRepPrimAPI_MakePrism
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.GeomAbs import GeomAbs_Circle
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_EDGE,TopAbs_FACE,TopAbs_REVERSED
from OCP.TopoDS import TopoDS,TopoDS_Compound
from OCP.BRep import BRep_Builder,BRep_Tool
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopLoc import TopLoc_Location
from OCP.STEPControl import STEPControl_Writer,STEPControl_AsIs
from OCP.IFSelect import IFSelect_RetDone

def compound(shapes):
 c=TopoDS_Compound();b=BRep_Builder();b.MakeCompound(c)
 for s in shapes:b.Add(c,s)
 return c
def transform(s,z=0,mirror=False):
 t=gp_Trsf()
 if mirror:t.SetMirror(gp_Ax2(gp_Pnt(0,0,0),gp_Dir(1,0,0)))
 s=BRepBuilderAPI_Transform(s,t,True).Shape() if mirror else s
 if z:
  t=gp_Trsf();t.SetTranslation(gp_Vec(0,0,z));s=BRepBuilderAPI_Transform(s,t,True).Shape()
 return s
def solid_text(faces,z,depth,mirror=False):return compound([transform(BRepPrimAPI_MakePrism(f,gp_Vec(0,0,depth)).Shape(),z,mirror) for f in faces])
def threshold(signal):
    """Separate the foreground ink from the black background with Otsu's method."""
    hist, edges = np.histogram(signal, bins=256, range=(0., 256.))
    total = hist.sum()
    weight = hist.cumsum()[:-1]
    moment = (hist * np.arange(256, dtype=float)).cumsum()
    valid = (weight > 0) & (weight < total)
    score = np.zeros_like(weight, dtype=float)
    score[valid] = (moment[-1]*weight[valid] - total*moment[:-1][valid])**2 / (
        weight[valid] * (total-weight[valid]))
    return float(edges[np.argmax(score)+1]) + .01


def polygon_area(points):
    x, y = points.T
    return .5 * np.dot(x, np.roll(y, -1) - np.roll(y, 1))


def polygon_wire(points, *, outer):
    if (polygon_area(points) > 0) != outer:
        points = points[::-1]
    polygon = BRepBuilderAPI_MakePolygon()
    for x, y in points:
        polygon.Add(gp_Pnt(float(x), float(y), 0.))
    polygon.Close()
    if not polygon.IsDone():
        raise ValueError("could not close the PDF contour")
    return polygon.Wire()


def face_signals(image_path, team):
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=float)
    red, green, blue = np.moveaxis(rgb, 2, 0)
    light = rgb @ np.array([.299, .587, .114])
    ink = np.maximum(red - (green+blue)/2, 0.) if team == "A" else (red+green+blue)/3
    return gaussian_filter(ink, .35), light


def ring_profile(signal, cutoff):
    h, w = signal.shape
    center = np.array([(w-1)/2, (h-1)/2])
    angle = np.linspace(0., 2*np.pi, 360, endpoint=False)
    directions = np.column_stack((np.cos(angle), np.sin(angle)))
    radii = np.arange(.33*min(h, w), .49*min(h, w), .08)
    for _ in range(2):
        pixels = center[None, None, :] + radii[None, :, None] * directions[:, None, :]
        ion = map_coordinates(signal, [pixels[..., 1], pixels[..., 0]], order=1) >= cutoff
        valid = np.flatnonzero(ion.any(axis=1))
        outer_ix = len(radii)-1-np.argmax(ion[valid, ::-1], axis=1)
        ends = pixels[valid, outer_ix]
        start = np.r_[center, np.median(radii[outer_ix])]
        center_x, center_y, outer = least_squares(
            lambda c: np.linalg.norm(ends-c[:2], axis=1)-c[2], start,
            loss="soft_l1", f_scale=.25).x
        center = np.array([center_x, center_y])
    thickness = []
    for row, end in zip(ion[valid], outer_ix, strict=True):
        start = end
        while start > 0 and row[start-1]:
            start -= 1
        if start > 0:
            thickness.append(radii[end]-radii[start]+.08)
    return center, outer, outer-float(np.median(thickness))


def smooth_polygon(points):
    """Smooth the scan outline at subpixel scale while leaving overall marks in place."""
    closed = np.vstack((points, points[0]))
    along = np.r_[0., np.linalg.norm(np.diff(closed, axis=0), axis=1).cumsum()]
    count = max(8, round(along[-1] / .40))
    at = np.linspace(0., along[-1], count, endpoint=False)
    uniform = np.column_stack([np.interp(at, along, axis) for axis in closed.T])
    return gaussian_filter1d(uniform, .78, axis=0, mode="wrap")


def normalize_ink(image, roi):
    black, ink = np.percentile(image[roi], [10, 94])
    return np.clip((image-black) / max(ink-black, 1.), 0., 1.)


def recover_strokes(signal, light, cutoff, center, inner, team):
    """Use luma edges and select a thinner trace by correlation at source resolution."""
    h, w = signal.shape
    y, x = np.indices(signal.shape)
    roi = np.hypot(x-center[0], y-center[1]) < inner-1.2
    chroma = normalize_ink(signal, roi)
    luminance = normalize_ink(light, roi)
    evidence = (.35*chroma + .65*luminance) if team == "A" else luminance
    sharp = evidence + 1.25*(evidence-gaussian_filter(evidence, .8))
    factor = 4
    gx = (np.arange(w*factor)+.5)/factor-.5
    gy = (np.arange(h*factor)+.5)/factor-.5
    xx, yy = np.meshgrid(gx, gy)
    radius = np.hypot(xx-center[0], yy-center[1])
    samples = map_coordinates(sharp, [yy, xx], order=3)
    previous = map_coordinates(signal, [yy, xx], order=3) >= cutoff
    before = previous.reshape(h, factor, w, factor).mean(axis=(1, 3))[roi].sum()
    if team == "A":
        lower, upper = .78, .87
    else:
        lower, upper = .88, .95
    choices = []
    for level in np.arange(.53, .81, .01):
        mask = (samples >= level) & (radius < inner-.34)
        reduced = mask.reshape(h, factor, w, factor).mean(axis=(1, 3))
        ratio = reduced[roi].sum()/before
        blurred = gaussian_filter(reduced, .65)
        similarity = float(np.corrcoef(blurred[roi], evidence[roi])[0, 1])
        deviation = max(lower-ratio, ratio-upper, 0.)
        choices.append((deviation, -similarity, level))
    level = min(choices)[2]
    contour = np.minimum(samples, level + (inner-.34-radius)*2.)
    return contour_generator(x=gx, y=gy, z=contour, fill_type="OuterOffset").filled(level, 3.)


def narrow_strokes(profile, width_reduction):
    if width_reduction == 0:
        return [profile]

    polygon = Polygon(profile[0], profile[1:])
    inset = polygon.buffer(-width_reduction / 2, quad_segs=8)
    inset = inset.simplify(.012, preserve_topology=True)
    return [[np.asarray(part.exterior.coords)[:-1],
             *(np.asarray(hole.coords)[:-1] for hole in part.interiors)]
            for part in get_parts(inset) if not part.is_empty]


def artwork_profile(image_path, team, artwork_diameter, width_reduction=0.):
    signal, light = face_signals(image_path, team)
    cutoff = threshold(signal)
    center, outer, inner = ring_profile(signal, cutoff)
    scale = artwork_diameter / (2*outer)
    outlines, offsets = recover_strokes(signal, light, cutoff, center, inner, team)
    faces = []
    profiles = []
    for path, offsets_for_path in zip(outlines, offsets, strict=True):
        points = (path-center) * [scale, -scale]
        paths = [points[a:b-1] for a,b in zip(offsets_for_path, offsets_for_path[1:])]
        outer, *holes = paths
        if len(outer) < 3 or abs(polygon_area(outer)) < .55:
            continue
        outer = smooth_polygon(outer)
        restored = [outer]
        for hole in holes:
            if len(hole) >= 3 and abs(polygon_area(hole)) >= .50:
                restored.append(smooth_polygon(hole))
        profiles.extend(narrow_strokes(restored, width_reduction))
    for outer, *holes in profiles:
        face = BRepBuilderAPI_MakeFace(polygon_wire(outer, outer=True), True)
        for hole in holes:
            face.Add(polygon_wire(hole, outer=False))
        shape = face.Face()
        if not BRepCheck_Analyzer(shape).IsValid():
            raise ValueError(f"invalid PDF contour: {image_path}")
        faces.append(shape)
    if not faces:
        raise ValueError(f"empty PDF ink mask: {image_path}")
    return faces, profiles, artwork_diameter/2, inner*scale


def mesh(s,linear=.14):
 BRepMesh_IncrementalMesh(s,linear,False,.16,True);e=TopExp_Explorer(s,TopAbs_FACE);out=[]
 while e.More():
  f=TopoDS.Face_s(e.Current());loc=TopLoc_Location();t=BRep_Tool.Triangulation_s(f,loc)
  if t:
   nodes=np.array([[v.X(),v.Y(),v.Z()] for v in [t.Node(i).Transformed(loc.Transformation()) for i in range(1,t.NbNodes()+1)]])
   tri=np.array([t.Triangle(i).Get() for i in range(1,t.NbTriangles()+1)])-1
   if f.Orientation()==TopAbs_REVERSED:tri=tri[:,::-1]
   out.extend(nodes[tri])
  e.Next()
 return np.array(out,dtype=np.float32)


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SIM = REPO / "sim/resources/chess"


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cylinder(radius, height, z=0.):
    axis = gp_Ax2(gp_Pnt(0, 0, z), gp_Dir(0, 0, 1))
    return BRepPrimAPI_MakeCylinder(axis, radius, height).Shape()


def ring(outer, inner, z, depth):
    return BRepAlgoAPI_Cut(cylinder(outer, depth, z),
                          cylinder(inner, depth+.04, z-.02)).Shape()


def export_artwork(folder, item, profiles, radii, body_rgb, ink_rgb, piece_radius):
    """Save a vector face and its antialiased 1024-pixel texture."""
    folder.mkdir(exist_ok=True)
    outer, inner = radii
    color = lambda rgb: "#{:02x}{:02x}{:02x}".format(*rgb)
    paths = []
    for profile in profiles:
        path = " ".join("M"+" L".join(f"{x:.3f},{-y:.3f}" for x, y in outline)+" Z"
                        for outline in profile)
        paths.append(f'<path fill="{color(ink_rgb)}" fill-rule="evenodd" d="{path}"/>')
    r = piece_radius
    text = (f'<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" '
            f'viewBox="{-r} {-r} {2*r} {2*r}">\n'
            f'<circle r="{r}" fill="{color(body_rgb)}"/>\n'
            f'<circle r="{(outer+inner)/2:.3f}" fill="none" stroke="{color(ink_rgb)}" '
            f'stroke-width="{outer-inner:.3f}"/>\n'+"\n".join(paths)+"\n</svg>\n")
    (folder / f'{item["id"]}.svg').write_text(text, encoding="utf-8")
    size = 2048
    image = Image.new("RGBA", (size, size))
    draw = ImageDraw.Draw(image)
    center, scale = size/2, size/(2*r)
    def circle(radius):
        return (center-radius*scale, center-radius*scale,
                center+radius*scale, center+radius*scale)
    draw.ellipse(circle(r), fill=tuple(body_rgb)+(255,))
    draw.ellipse(circle(outer), fill=tuple(ink_rgb)+(255,))
    draw.ellipse(circle(inner), fill=tuple(body_rgb)+(255,))
    for outline, *holes in profiles:
        draw.polygon([(center+x*scale, center-y*scale) for x, y in outline],
                     fill=tuple(ink_rgb)+(255,))
        for hole in holes:
            draw.polygon([(center+x*scale, center-y*scale) for x, y in hole],
                         fill=tuple(body_rgb)+(255,))
    image.resize((1024, 1024), Image.Resampling.LANCZOS).save(folder / f'{item["id"]}.png')


def to_mesh(shape, color=None):
    vertices = mesh(shape, .10).reshape(-1, 3)
    surface = trimesh.Trimesh(vertices=vertices, faces=np.arange(len(vertices)).reshape(-1, 3),
                              process=True)
    if color is not None:
        surface.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=color, metallicFactor=0., roughnessFactor=.57))
    return surface


def export_step_stl(shape, surface, step_path, stl_path):
    writer = STEPControl_Writer()
    if writer.Transfer(shape, STEPControl_AsIs) != IFSelect_RetDone:
        raise RuntimeError(f"CAD transfer failed: {step_path}")
    if writer.Write(str(step_path)) != IFSelect_RetDone:
        raise RuntimeError(f"STEP export failed: {step_path}")
    step_path.write_bytes(b"\n".join(line.rstrip() for line in step_path.read_bytes().splitlines()) + b"\n")
    if not surface.is_watertight or not surface.is_volume or not np.isfinite(surface.vertices).all():
        raise RuntimeError(f"invalid printable mesh: {stl_path}")
    surface.export(stl_path)


def preview(config, glyphs, ring_sizes, font_path):
    # Top view drawn from the same glyph triangles as the 3D exports.
    width, height = 1680, 700
    image = Image.new("RGB", (2*width, 2*height), "#eaf0f4")
    d = ImageDraw.Draw(image)
    font = lambda size: ImageFont.truetype(str(font_path), size*2)
    d.text((70, 42), "智能車光電象棋  ·  區賽規格", font=font(36), fill="#172c40")
    d.text((70, 142), "70 號雙面棋   /   Ø 67 mm × 17 mm   /   甲紅、乙白，共 32 枚",
           font=font(20), fill="#3e556a")
    for i, piece in enumerate(config["types"]):
        row, col = divmod(i, 7)
        x, y = 130 + 236*col, 228 + 235*row
        radius = 79
        d.ellipse(((x-radius)*2, (y+11-radius)*2, (x+radius)*2, (y+11+radius)*2), fill="#11171d")
        d.ellipse(((x-radius)*2, (y-radius)*2, (x+radius)*2, (y+radius)*2),
                  fill=tuple(config["colors"]["body"][:3]), outline="#485158", width=2)
        color = tuple(config["colors"]["red" if piece["team"] == "A" else "white"][:3])
        scale = radius / (config["diameter_mm"]/2)
        outer, inner = ring_sizes[piece["id"]]
        ring_px = outer * scale
        d.ellipse(((x-ring_px)*2, (y-ring_px)*2, (x+ring_px)*2, (y+ring_px)*2),
                  outline=color, width=round(2*(outer-inner)*scale))
        for tri in glyphs[piece["id"]]:
            if np.all(tri[:, 2] > config["height_mm"]*.5):
                d.polygon([((x+u*scale)*2, (y-v*scale)*2) for u, v in tri[:, :2]], fill=color)
        d.text((2*x, 2*(y+108)), f'{piece["character"]}  ×{piece["count"]}  ·  {piece["points"]} 分',
               font=font(19), anchor="mm", fill="#25394d")
    reductions = config.get("stroke_width_reduction_mm", {})
    subtitle = "組委會 PDF 字形  /  正反面同一圖樣"
    if reductions:
        red = reductions.get("red", 0.)
        white = reductions.get("white", 0.)
        subtitle += f"  /  筆畫收窄  紅 {red:.2f} mm、白 {white:.2f} mm"
    d.text((70, 1293), subtitle, font=font(18), fill="#3e556a")
    image.resize((width, height), Image.Resampling.LANCZOS).save(HERE/"preview.png")


def create_model(font_path):
    config = json.loads((HERE/"chess.json").read_text(encoding="utf-8"))
    radius, height = config["diameter_mm"]/2, config["height_mm"]
    # Build the edge list from the same topological solid handed to the fillet.
    blank = cylinder(radius, height)
    fillet = BRepFilletAPI_MakeFillet(blank)
    edges = TopExp_Explorer(blank, TopAbs_EDGE)
    while edges.More():
        edge = TopoDS.Edge_s(edges.Current())
        if BRepAdaptor_Curve(edge).GetType() == GeomAbs_Circle:
            fillet.Add(config["edge_fillet_mm"], edge)
        edges.Next()
    body = fillet.Shape()
    relief = config["letter_engraving_depth_mm"]
    ink_depth = config["letter_ink_thickness_mm"]
    for directory in ("glb", "step", "stl"):
        (HERE/directory).mkdir(exist_ok=True)
    SIM.mkdir(parents=True, exist_ok=True)
    all_meshes, parts_by_id, glyphs, ring_sizes = {}, {}, {}, {}
    for item in config["types"]:
        source = HERE / "faces" / f'{item["id"]}.png'
        team_color = "red" if item["team"] == "A" else "white"
        faces, paths, outer, inner = artwork_profile(
            source, item["team"], config["artwork_outer_diameter_mm"],
            config.get("stroke_width_reduction_mm", {}).get(team_color, 0.),
        )
        ring_sizes[item["id"]] = outer, inner
        export_artwork(HERE/"faces_hd", item, paths, (outer, inner),
                       config["colors"]["body"][:3], config["colors"][team_color][:3], radius)
        cutter = compound([solid_text(faces, height-relief, relief+.06),
                           solid_text(faces, -.06, relief+.06, True),
                           ring(outer, inner, height-relief, relief+.06),
                           ring(outer, inner, -.06, relief+.06)])
        shape = BRepAlgoAPI_Cut(body, cutter).Shape()
        if not BRepCheck_Analyzer(shape).IsValid():
            raise RuntimeError(f"invalid CAD for {item['id']}")
        ink = compound([solid_text(faces, height-relief, ink_depth),
                        solid_text(faces, relief-ink_depth, ink_depth, True)])
        border = compound([ring(outer, inner, height-relief, ink_depth),
                           ring(outer, inner, relief-ink_depth, ink_depth)])
        glyphs[item["id"]] = mesh(ink)
        scene = trimesh.Scene()
        parts = {}
        bounds = None
        for part, solid, color in (("body", shape, "body"), ("ink", ink, team_color),
                                   ("ring", border, team_color)):
            material = config["colors"][color]
            surface = to_mesh(solid, material)
            if part == "body":
                export_step_stl(shape, surface, HERE/item["step"], HERE/item["stl"])
            surface.apply_scale(.001)
            scene.add_geometry(surface.copy(), node_name=part, geom_name=part)
            surface.apply_translation([0., 0., -height*.0005])
            if part == "body":
                bounds = surface.bounds.tolist()
            index = len(all_meshes)
            file = SIM/f"visual_{index:02}.obj"
            file.write_text(trimesh.exchange.obj.export_obj(surface, include_color=False).rstrip() + "\n")
            all_meshes[str(index)] = dict(file=file.name, name=f'{item["id"]}_{part}',
                rgba=[v/255 for v in material], vertices=len(surface.vertices), triangles=len(surface.faces),
                bounds_m=surface.bounds.tolist(), sha256=hashlib.sha256(file.read_bytes()).hexdigest())
            parts[part] = index
        scene.export(HERE/item["glb"])
        parts_by_id[item["id"]] = parts, bounds
        print(f"Exported {item['id']}", flush=True)

    to_mesh(cylinder(radius, height)).export(HERE/"collision.stl")
    groups = [[(item, i) for item in config["types"] if item["team"] == team
               for i in range(1, item["count"]+1)] for team in ("A", "B")]
    pieces = []
    for pair in zip(*groups, strict=True):
        for item, i in pair:
            parts, bounds = parts_by_id[item["id"]]
            pieces.append(dict(instance_id=len(pieces)+1, source_piece_id=f'{item["id"]}_{i}',
                side=item["side"], team=item["team"], character=item["character"], points=item["points"],
                **{"class": "red" if item["team"] == "A" else "white"}, mesh_parts=parts,
                visual_bounds_m=bounds, collision_radius_m=radius*.001,
                collision_half_height_m=height*.0005))
    assert len(pieces) == config["piece_count"] == 32
    save_json(SIM/"chess_manifest.json", dict(
        units="m", body_frame_origin="cylinder axis at mid-height", pieces=pieces, meshes=all_meshes))
    used_files = {row["file"] for row in all_meshes.values()}
    for stale in SIM.glob("visual_*.obj"):
        if stale.name not in used_files:
            stale.unlink()
    preview(config, glyphs, ring_sizes, font_path)
    for directory in ("glb", "step", "stl"):
        for superseded in (HERE/directory).glob(f"黑_*.{directory}"):
            superseded.unlink()


def update_asset_manifest():
    path = REPO/"assets/manifests/models.json"
    data = json.loads(path.read_text())
    data["source_selection"]["chess"] = "2026-11-17 competition notice and attachment 1"
    keep = [item for item in data["files"]
            if not item["path"].startswith(("assets/chess/", "sim/resources/chess/"))]
    paths = [HERE/"chess.json", HERE/"collision.stl", HERE/"preview.png",
             *HERE.glob("step/*.step"), *HERE.glob("stl/*.stl"), *HERE.glob("glb/*.glb"),
             *(HERE/"faces").glob("*.png"), HERE/"faces/source.json",
             *(HERE/"faces_hd").glob("*.svg"), *(HERE/"faces_hd").glob("*.png"),
             SIM/"chess_manifest.json", *SIM.glob("visual_*.obj")]
    data["files"] = sorted(keep + [dict(path=str(p.relative_to(REPO)), bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths], key=lambda x: x["path"])
    save_json(path, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-font", type=Path, required=True, help="CJK font for preview labels only")
    args = parser.parse_args()
    create_model(args.label_font)
    update_asset_manifest()
