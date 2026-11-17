"""Generate the manuscript's OSM case section from independently checked data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results" / "confirmation"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verification = json.loads((RESULTS / "verification.json").read_text(encoding="utf-8"))
    results_path = RESULTS / "analysis.json"
    if (verification["result"] != "PASS" or verification["episodes"] != 450 or
            verification["analysis_sha256"] != hashlib.sha256(results_path.read_bytes()).hexdigest()):
        raise RuntimeError("Confirmation results are not fully verified")
    analysis = json.loads(results_path.read_text(encoding="utf-8"))
    network = json.loads((DATA / "network_summary.json").read_text(encoding="utf-8"))
    cells = {(r["profile"], r["method"]): r for r in analysis["cells"]}
    diffs = {(r["profile"], r["control"]): r for r in analysis["comparisons"]}
    profiles = (("coupled_primary", "Primary"), ("edge_dominant", "Edge limited"),
                ("tighter_deadlines", "Tight deadlines"))
    methods = ("rosr", "e2e_resource_density", "joint_lns", "coalition_value", "lit_dpso")

    def delta(profile, control):
        rec = diffs[profile, control]["value_difference_pp"]
        return f'{rec["mean"]:+.2f} [{rec["low95"]:+.2f},{rec["high95"]:+.2f}]'

    table_rows = []
    for profile, name in profiles:
        means = [f'{cells[profile, method]["mean_value_pct"]:.2f}' for method in methods]
        w = diffs[profile, "e2e_resource_density"]
        table_rows.append(" & ".join([name, *means, delta(profile, "e2e_resource_density"),
                                      f'{w["wins"]}/{w["ties"]}/{w["losses"]}']) + r'\\')
    primary = cells["coupled_primary", "rosr"]
    edge = cells["edge_dominant", "rosr"]
    edge_density = cells["edge_dominant", "e2e_resource_density"]
    tight = cells["tighter_deadlines", "rosr"]
    primary_joint = cells["coupled_primary", "joint_lns"]
    edge_joint = cells["edge_dominant", "joint_lns"]
    tight_joint = cells["tighter_deadlines", "joint_lns"]
    paragraph = rf'''
\subsection{{Hefei road-network case study}}\label{{sec:osm}}

We examine the policy on a directed road topology in Hefei, China (Figure~\ref{{fig:osm}}), while retaining synthetic task and vehicle attributes. An OpenStreetMap snapshot retrieved on 12 November 2026 was filtered for drivable roads and converted with OSMnx \citep{{osmfoundation2026,boeing2025osmnx}}; the drive filter excludes service and private roads. Its largest strongly connected component retains {network["component_nodes"]:,} of {network["original_nodes"]:,} nodes and {network["component_directed_edges"]:,} of {network["original_directed_edges"]:,} directed links. The sampling rectangle spans 31.815--31.842\(^\circ\)N and 117.225--117.255\(^\circ\)E, with a 0.01-degree routing buffer. The original 100 grid locations are snapped to distinct road nodes; median and maximum snap distances are {network["snap_m_median"]:.1f} and {network["snap_m_max"]:.1f}\,m.

For each ordered pair, travel uses directed shortest-road distance divided by a constant vehicle speed. One original distance unit is mapped to 300\,m, giving synthetic speeds of 27--45\,km/h from the original speed distribution; the mapping retains the original task generator, service times and deadlines. It represents neither measured traffic nor road-specific speeds, signals or turn restrictions. All methods use the same road-time matrix and fixed-upload contract. Independent SciPy and NetworkX shortest-path calculations agree on all 10,000 ordered pairs. Thirty new paired seeds per workload (20480000--20480029) give 450 audited episodes across the five methods in Table~\ref{{tab:osm}}; the same seeds are reused across related workloads. The two pilot seeds were disjoint from this comparison block. Physical and action-log audits are replayed from the retained traces against the independently computed road-distance matrix. OpenAI Codex (OpenAI) assisted with the case preprocessing, evaluation scripts and Figure~\ref{{fig:osm}}; the scripts, inputs and outputs are retained for reproducibility.

\begin{{figure}}[pos=H]\centering
\includegraphics[width=0.92\linewidth]{{figures/fig9_osm.pdf}}
\caption{{Hefei drivable road network and the 100 snapped candidate locations for synthetic tasks and vehicle starts. The dashed rectangle denotes the original sampling area; snapping can place a location just outside it. Map data \copyright\ OpenStreetMap contributors, ODbL (\url{{https://www.openstreetmap.org/copyright}}).}}\label{{fig:osm}}
\end{{figure}}

\begin{{table}}[pos=H]\centering\footnotesize\setlength{{\tabcolsep}}{{2.5pt}}
\caption{{Hefei road-network case: completed value (\%) and paired RCR-minus-Density effects (percentage points, unadjusted 95\% Student intervals). W/T/L counts are against Density over 30 paired streams.}}\label{{tab:osm}}
\begin{{tabular}}{{lrrrrrlc}}\toprule
Workload & RCR & Density & Joint DR & Coal.-V & DPSO & RCR $-$ Density & W/T/L\\\midrule
'''+"\n".join(table_rows)+rf'''
\bottomrule\end{{tabular}}\end{{table}}

RCR exceeds Density by {delta("coupled_primary", "e2e_resource_density")} percentage points in the primary workload, {delta("edge_dominant", "e2e_resource_density")} under edge limitation and {delta("tighter_deadlines", "e2e_resource_density")} under tight deadlines. The win/tie/loss counts show that these average gains do not imply dominance on every stream. In the primary workload, RCR minus Coalition-V is {delta("coupled_primary", "coalition_value")} points and RCR minus DPSO is {delta("coupled_primary", "lit_dpso")}. RCR minus Joint DR is {delta("coupled_primary", "joint_lns")}, {delta("edge_dominant", "joint_lns")} and {delta("tighter_deadlines", "joint_lns")} points, respectively; the first two intervals include zero. All intervals are descriptive and unadjusted for multiple comparisons.

The value--waste tradeoff also appears on this network. Under edge limitation, downstream loss is {edge["mean_downstream_pct"]:.2f}\% for RCR and {edge_density["mean_downstream_pct"]:.2f}\% for Density, while wasted acquisition time is {edge["mean_wasted_minutes"]:.2f} and {edge_density["mean_wasted_minutes"]:.2f} vehicle minutes per episode. Mean planning times for RCR and Joint DR are {primary["mean_planning_seconds"]:.2f}/{primary_joint["mean_planning_seconds"]:.2f}\,s in the primary workload, {edge["mean_planning_seconds"]:.2f}/{edge_joint["mean_planning_seconds"]:.2f}\,s with limited edge capacity and {tight["mean_planning_seconds"]:.2f}/{tight_joint["mean_planning_seconds"]:.2f}\,s with tight deadlines. Times were measured with three episode workers and a wall-clock search limit; they can vary with load and do not imply a hard response-time guarantee. This single network adds topological diversity to the grid tests but does not validate real demand, travel-time distributions or other cities.
'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(paragraph.strip() + "\n", encoding="utf-8")
    print(f"Wrote {args.output} from {verification['episodes']} audited episodes")


if __name__ == "__main__":
    main()
