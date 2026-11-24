"""Render dimensioned views of the printable pusher mesh."""
from pathlib import Path
import argparse
from hashlib import sha256
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def refresh_manifest(params):
    file = REPO/"assets/manifests/models.json"
    manifest = json.loads(file.read_text())
    manifest["source_selection"]["pusher"] = (
        f'{params["max_width"]:g} mm arc pusher, unchanged M3 mounting interface')
    paths = [HERE/name for name in ("arcs.json", "parameters.json", "reinforcement.json",
                                   "pusher.step", "pusher.stl", "pusher.obj", "preview.png",
                                   "build_pusher.py", "build_simulation.py", "render_preview.py")]
    paths.append(REPO/"assets/robot/simulation.json")
    records = {}
    for p in paths:
        data = p.read_bytes()
        records[str(p.relative_to(REPO))] = dict(path=str(p.relative_to(REPO)), bytes=len(data),
                                                sha256=sha256(data).hexdigest())
    keep = [row for row in manifest["files"] if row["path"] not in records]
    manifest["files"] = sorted(keep+list(records.values()), key=lambda row: row["path"])
    file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")


def render(font_path, params):
    mesh = trimesh.load_mesh(HERE/"pusher.stl")
    xyz = mesh.triangles
    font = FontProperties(fname=font_path)
    ink, faint = "#1b3448", "#5d7382"
    teal = np.array([.16, .62, .65])
    fig = plt.figure(figsize=(13.8, 7.4), facecolor="#edf3f6")
    width, depth = params["max_width"], params["forward_extent_from_hole_line"]
    fig.text(.065,.935,f"三段圆弧推板 · {width:g} mm",fontproperties=font,fontsize=29,color=ink)
    fig.text(.066,.88,
             f'两侧内弧 Ø{params["side_inner_diameter_mm"]:g} mm'
             f'   /   单侧圆心角 {params["side_arc_sweep_deg"]:.2f}°',
             fontproperties=font,fontsize=15,color=faint)
    figs = fig.add_gridspec(1,2,left=.055,right=.955,bottom=.14,top=.81,width_ratios=(1.13,1))
    top = fig.add_subplot(figs[0],facecolor="#edf3f6")
    for above, color in ((xyz[mesh.face_normals[:,2]>.95], "#76b7ba"),
                         (xyz[np.min(xyz[:,:,2],axis=1)>26.99], "#227b80")):
        top.add_collection(PolyCollection(above[:,:,:2],facecolors=color,edgecolors="none"))
    for x in (-width/2, width/2):
        top.plot([x,x],[65,depth+15],color=faint,linewidth=.75,linestyle="--")
    top.annotate("",xy=(-width/2,depth+10),xytext=(width/2,depth+10),
                 arrowprops={"arrowstyle":"<->","color":ink,"linewidth":1.2})
    top.text(0,depth+13,f"{width:g} mm",ha="center",va="bottom",fontsize=15,color=ink)
    for x in (-9.5,9.5):
        top.add_patch(Circle((x,0),1.5,facecolor="#edf3f6",edgecolor=ink,lw=.6))
    top.annotate("2 × Ø3 mm，孔距 19 mm",(9.5,0),(37,-13),fontproperties=font,
                 fontsize=12,color=ink,arrowprops={"arrowstyle":"->","color":ink})
    top.set(xlim=(-165,165),ylim=(-22,126),aspect="equal")
    top.axis("off")
    top.set_title("俯视图",fontproperties=font,fontsize=16,color=ink)
    iso = fig.add_subplot(figs[1],projection="3d",facecolor="#edf3f6")
    light = np.array([-.2,.35,.91])
    shade = np.clip(.58+.40*(mesh.face_normals@light),.35,1.)
    iso.add_collection3d(Poly3DCollection(xyz,facecolors=shade[:,None]*teal,
                         edgecolors="none",antialiased=False))
    iso.set(xlim=(-162,162),ylim=(-15,110),zlim=(0,35))
    iso.set_box_aspect((324,125,60))
    iso.view_init(elev=36,azim=-67)
    iso.set_axis_off()
    iso.set_title("轴测图",fontproperties=font,fontsize=16,color=ink,pad=0)
    footer = (f'推壁高 {params["push_height"]:g} mm · 壁厚 {params["wall_thickness_mm"]:g} mm'
              f' · 离地 {params["nominal_ground_clearance"]:g} mm'
              f' · 孔周保留 {params["hole_region_thickness_mm"]:g} mm')
    fig.text(.065,.065,footer,
             fontproperties=font,fontsize=14,color=ink)
    fig.savefig(HERE/"preview.png",dpi=130,facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-font",type=Path,required=True)
    args=parser.parse_args()
    params = json.loads((HERE/"parameters.json").read_text())
    render(args.label_font, params)
    refresh_manifest(params)
