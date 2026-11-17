"""Download and prepare a reproducible, directed OSM road-network instance.

The 100 task/vehicle locations are snapped from a fixed 10-by-10 set of
points to the largest strongly connected drivable component. Two independent
shortest-path implementations are compared before the matrices are saved.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import networkx as nx
import numpy as np
import osmnx as ox
from pyproj import Transformer
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ENDPOINT = "https://api.openstreetmap.org/api/0.6/map"
# Longitude first in the location grid; the outer box adds a routing buffer.
GRID = {"west": 117.225, "east": 117.255, "south": 31.815, "north": 31.842}
BUFFER = 0.010
UNIT_M = 300.0
USER_AGENT = "VCS-E2E-academic-network-case/1.0 (OpenStreetMap research)"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    existing = DATA / "network_summary.json"
    if existing.exists():
        summary = json.loads(existing.read_text(encoding="utf-8"))
        for filename, field in (("hefei_bbox.osm", "raw_osm_sha256"),
                                ("hefei_drive.osm", "filtered_osm_sha256"),
                                ("hefei_drive.graphml", "graphml_sha256"),
                                ("distance_m.npy", "matrix_sha256"),
                                ("audit_distance_m.npy", "audit_matrix_sha256"),
                                ("anchors.json", "anchors_sha256")):
            if sha(DATA / filename) != summary[field]:
                raise RuntimeError(f"Prepared OSM evidence changed: {filename}")
        print("Existing frozen OSM network and matrices verified; no files overwritten.", flush=True)
        return
    box = (GRID["south"] - BUFFER, GRID["west"] - BUFFER,
           GRID["north"] + BUFFER, GRID["east"] + BUFFER)
    # OSM map API uses west,south,east,north. Keep the raw response so the
    # exact snapshot can be reproduced without relying on a live service.
    bbox = (box[1], box[0], box[3], box[2])
    query = urlencode({"bbox": ",".join(map(str, bbox))})
    xml_path = DATA / "hefei_bbox.osm"
    request_meta = DATA / "download.json"
    if not xml_path.exists():
        url = ENDPOINT + "?" + query
        request = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=180) as response:
            body = response.read()
            status = response.status
            http_date = response.headers.get("Date")
        if status != 200 or len(body) < 1000 or b"<osm " not in body[:2000]:
            raise RuntimeError(f"Unexpected Overpass response: HTTP {status}, {len(body)} bytes")
        root = ET.fromstring(body)
        part = xml_path.with_suffix(".osm.part")
        part.write_bytes(body)
        part.replace(xml_path)
        request_meta.write_text(json.dumps({
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "http_date": http_date, "endpoint": ENDPOINT, "query": query,
            "raw_osm_sha256": sha(xml_path), "bytes": len(body),
            "source": "OpenStreetMap contributors", "license": "ODbL-1.0",
            "license_url": "https://www.openstreetmap.org/copyright",
        }, indent=2) + "\n", encoding="utf-8")
    else:
        meta = json.loads(request_meta.read_text(encoding="utf-8"))
        if meta["query"] != query or meta["raw_osm_sha256"] != sha(xml_path):
            raise RuntimeError("Existing OSM snapshot does not match query or recorded hash")

    # Apply OSMnx 2.1.1's documented drive filter to the local XML. This
    # excludes footpaths, private access and roads closed to motor vehicles.
    source = ET.parse(xml_path).getroot()
    excluded = re.compile("abandoned|bridleway|bus_guideway|construction|corridor|cycleway|elevator|escalator|footway|no|path|pedestrian|planned|platform|proposed|raceway|razed|rest_area|service|services|steps|track")
    bad_service = re.compile("alley|driveway|emergency_access|parking|parking_aisle|private")
    ways = []
    used_nodes = set()
    for way in source.findall("way"):
        tags = {t.attrib["k"]: t.attrib["v"] for t in way.findall("tag")}
        highway = tags.get("highway")
        if (not highway or re.search("yes", tags.get("area", "")) or
                re.search("private", tags.get("access", "")) or excluded.search(highway) or
                re.search("no", tags.get("motor_vehicle", "")) or
                re.search("no", tags.get("motorcar", "")) or bad_service.search(tags.get("service", ""))):
            continue
        ways.append(way)
        used_nodes.update(nd.attrib["ref"] for nd in way.findall("nd"))
    nodes = [node for node in source.findall("node") if node.attrib["id"] in used_nodes]
    if len(nodes) != len(used_nodes) or len(ways) < 100:
        raise RuntimeError(f"Incomplete or unexpectedly small road extract: {len(nodes)} nodes, {len(ways)} ways")
    filtered = ET.Element("osm", {"version": "0.6", "generator": "VCS-E2E-OSM-drive-filter"})
    filtered.extend(nodes)
    filtered.extend(ways)
    filtered_path = DATA / "hefei_drive.osm"
    ET.ElementTree(filtered).write(filtered_path, encoding="utf-8", xml_declaration=True)
    # Retain intermediate OSM geometry nodes as eligible road locations:
    # simplifying to intersections alone can collapse nearby task locations.
    graph = ox.graph_from_xml(filtered_path, bidirectional=False, simplify=False, retain_all=True)
    n_before, m_before = len(graph), graph.number_of_edges()
    component = max(nx.strongly_connected_components(graph), key=len)
    graph = graph.subgraph(component).copy()
    projected = ox.project_graph(graph, to_crs="EPSG:32650")
    # OSMnx's conversion preserves the shortest parallel edge instead of
    # summing parallel edges as a generic adjacency-matrix conversion might.
    directed = ox.convert.to_digraph(projected, weight="length")
    node_ids = list(directed.nodes)
    node_index = {node: i for i, node in enumerate(node_ids)}
    xy = np.array([(directed.nodes[n]["x"], directed.nodes[n]["y"]) for n in node_ids])
    tree = cKDTree(xy)
    convert = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
    anchors = []
    for ix, lon in enumerate(np.linspace(GRID["west"], GRID["east"], 10)):
        for iy, lat in enumerate(np.linspace(GRID["south"], GRID["north"], 10)):
            x, y = convert.transform(float(lon), float(lat))
            offset, pos = tree.query([x, y])
            node = node_ids[int(pos)]
            anchors.append({"grid_x": ix, "grid_y": iy, "target_lon": float(lon),
                            "target_lat": float(lat), "node": int(node),
                            "snap_m": float(offset), "x_m": float(xy[pos, 0]),
                            "y_m": float(xy[pos, 1])})
    if len({a["node"] for a in anchors}) != 100:
        raise RuntimeError("Snapping produced duplicate locations; inspect network and protocol")

    positions = [node_index[a["node"]] for a in anchors]
    adjacency = nx.to_scipy_sparse_array(directed, nodelist=node_ids, weight="length", format="csr", dtype=float)
    # scipy's compiled csgraph implementation expects 32-bit sparse indices.
    adjacency.indices = adjacency.indices.astype(np.int32)
    adjacency.indptr = adjacency.indptr.astype(np.int32)
    dist = np.asarray(dijkstra(adjacency, directed=True, indices=positions))[:, positions]
    if dist.shape != (100, 100) or not np.isfinite(dist).all():
        raise RuntimeError("Some snapped road-network locations are unreachable")

    # Independent graph traversal to cross-check every OD entry. This matrix
    # is used by the action audit rather than the planner's lookup matrix.
    audit_dist = np.empty_like(dist)
    for i, anchor in enumerate(anchors):
        reachable = nx.single_source_dijkstra_path_length(directed, anchor["node"], weight="length")
        audit_dist[i] = [reachable[a["node"]] for a in anchors]
    max_error = float(np.max(np.abs(audit_dist - dist)))
    if max_error > 1e-5:
        raise RuntimeError(f"Independent shortest-path disagreement: {max_error} m")
    triangle_violation = max(float(np.max(dist - (dist[:, k, None] + dist[k, None, :]))) for k in range(100))
    if triangle_violation > 1e-5 or np.max(np.abs(np.diag(dist))) > 1e-8:
        raise RuntimeError("Distance matrix fails shortest-path metric conditions")

    matrix_path, audit_path = DATA / "distance_m.npy", DATA / "audit_distance_m.npy"
    np.save(matrix_path, dist)
    np.save(audit_path, audit_dist)
    (DATA / "anchors.json").write_text(json.dumps(anchors, indent=2) + "\n", encoding="utf-8")
    ox.save_graphml(projected, DATA / "hefei_drive.graphml")
    offsets = np.array([a["snap_m"] for a in anchors])
    off_diagonal = dist[~np.eye(100, dtype=bool)]
    summary = {
        "grid_bbox_lon_lat": GRID, "query_bbox_s_w_n_e": box, "grid_points": 100,
        "coordinate_system": "EPSG:32650", "distance_unit_m": UNIT_M,
        "original_nodes": n_before, "original_directed_edges": m_before,
        "component_nodes": len(projected), "component_directed_edges": projected.number_of_edges(),
        "unique_snapped_nodes": len({a["node"] for a in anchors}),
        "snap_m_median": float(np.median(offsets)), "snap_m_max": float(np.max(offsets)),
        "road_distance_m_median": float(np.median(off_diagonal)),
        "road_distance_m_p90": float(np.percentile(off_diagonal, 90)),
        "max_shortest_path_disagreement_m": max_error,
        "max_triangle_violation_m": triangle_violation,
        "max_directional_asymmetry_m": float(np.max(np.abs(dist - dist.T))),
        "graphml_sha256": sha(DATA / "hefei_drive.graphml"),
        "matrix_sha256": sha(matrix_path), "audit_matrix_sha256": sha(audit_path),
        "anchors_sha256": sha(DATA / "anchors.json"),
        "raw_osm_sha256": sha(xml_path), "filtered_osm_sha256": sha(filtered_path),
        "drive_filter": ox._overpass._get_network_filter("drive"), "osmnx_version": ox.__version__,
        "networkx_version": nx.__version__, "numpy_version": np.__version__,
        "note": "Static road lengths and constant per-vehicle speeds; no live traffic or turn-delay calibration.",
    }
    (DATA / "network_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
