"""Render the OSM study area and sampled task/vehicle locations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
import osmnx as ox
from pyproj import Transformer

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, help="Optional vector PDF output for the manuscript")
    args = parser.parse_args()
    info = json.loads((DATA / "network_summary.json").read_text(encoding="utf-8"))
    anchors = json.loads((DATA / "anchors.json").read_text(encoding="utf-8"))
    graph = ox.load_graphml(DATA / "hefei_drive.graphml")

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(7.0, 4.0), facecolor="white")
    ax = fig.add_axes([0.02, 0.05, 0.67, 0.90])
    annotation = fig.add_axes([0.69, 0.08, 0.30, 0.84])
    annotation.axis("off")

    segments, seen = [], set()
    for u, v in graph.edges():
        key = tuple(sorted((u, v)))
        if u == v or key in seen:
            continue
        seen.add(key)
        segments.append([(graph.nodes[u]["x"], graph.nodes[u]["y"]),
                         (graph.nodes[v]["x"], graph.nodes[v]["y"])])
    ax.add_collection(LineCollection(segments, colors="#c5cbd0", linewidths=0.5, zorder=1))

    box = info["grid_bbox_lon_lat"]
    transform = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
    corners = [transform.transform(lon, lat) for lon, lat in (
        (box["west"], box["south"]), (box["east"], box["south"]),
        (box["east"], box["north"]), (box["west"], box["north"]))]
    ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor="#b1653f",
                         linewidth=0.9, linestyle=(0, (4, 3)), zorder=2))
    ax.scatter([a["x_m"] for a in anchors], [a["y_m"] for a in anchors],
               s=12, c="#137582", edgecolors="white", linewidths=0.35, zorder=3)
    xs, ys = zip(*corners)
    ax.set_xlim(min(xs) - 650, max(xs) + 650)
    ax.set_ylim(min(ys) - 650, max(ys) + 650)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    x0, y0 = min(xs) - 450, min(ys) - 460
    ax.plot([x0, x0 + 1000], [y0, y0], color="#30383c", lw=2.2, solid_capstyle="butt", zorder=5)
    ax.text(x0 + 500, y0 + 65, "1 km", ha="center", va="bottom", fontsize=7, color="#30383c")
    ax.annotate("N", xy=(max(xs)+430, max(ys)+380), xytext=(max(xs)+430, max(ys)+80),
                arrowprops={"arrowstyle":"-|>", "color":"#30383c", "lw":0.8},
                ha="center", va="top", fontsize=8, color="#30383c")

    annotation.text(0.0, 0.96, "Hefei road network", fontsize=10, fontweight="bold", color="#20292d",
                    ha="left", va="top", transform=annotation.transAxes)
    annotation.plot([0.0, 0.16], [0.80, 0.80], color="#b1653f", lw=1, linestyle=(0, (4, 3)),
                    transform=annotation.transAxes, clip_on=False)
    annotation.text(0.20, 0.80, "Sampling area", va="center", fontsize=8, transform=annotation.transAxes)
    annotation.scatter([0.08], [0.68], s=18, c="#137582", edgecolors="white", linewidths=0.4,
                       transform=annotation.transAxes, clip_on=False)
    annotation.text(0.20, 0.68, "Candidate locations", va="center", fontsize=8, transform=annotation.transAxes)
    annotation.plot([0.0, 0.16], [0.56, 0.56], color="#adb8bf", lw=1,
                    transform=annotation.transAxes, clip_on=False)
    annotation.text(0.20, 0.56, "Drivable roads", va="center", fontsize=8, transform=annotation.transAxes)
    annotation.text(0.0, 0.40, f'{info["component_nodes"]:,} road nodes\n'
                    f'{info["component_directed_edges"]:,} directed links\n'
                    f'{info["grid_points"]} candidate locations',
                    linespacing=1.6, fontsize=8, color="#4b575d", va="top", transform=annotation.transAxes)
    annotation.text(0.0, 0.06, "Map data © OpenStreetMap\ncontributors · ODbL",
                    fontsize=6.5, color="#5b666b", va="bottom", transform=annotation.transAxes)
    out = DATA / "hefei_network"
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=220)
    if args.pdf:
        args.pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.pdf, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
