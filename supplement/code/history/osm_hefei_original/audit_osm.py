"""OSM adapter of the frozen physical and action-log audit."""
from __future__ import annotations
import model
from case_contract import audit_travel_time


def check_episode(summary, rows, logs):
    by_vehicle = {}
    edge_intervals = []
    for r in rows:
        if r["status"] not in ("completed", "expired"):
            raise AssertionError(f"nonterminal final task {r['task_id']}: {r['status']}")
        if r["status"] == "completed" and (r["edge_finish"] is None or r["edge_finish"] > r["deadline"] + model.EPS):
            raise AssertionError("late task counted as completed")
        if r["capture_start"] is not None:
            if r["capture_finish"] is None or r["capture_finish"] < r["capture_start"]:
                raise AssertionError("invalid vehicle interval")
            by_vehicle.setdefault(r["vehicle_id"], []).append((r["capture_start"], r["capture_finish"]))
        if r["edge_start"] is not None:
            if r["edge_finish"] is None or r["edge_finish"] < r["edge_start"]:
                raise AssertionError("invalid edge interval")
            edge_intervals.append((r["edge_start"], r["edge_finish"]))
    for intervals in [*by_vehicle.values(), edge_intervals]:
        intervals.sort()
        for a, b in zip(intervals, intervals[1:]):
            if a[1] > b[0] + model.EPS:
                raise AssertionError(f"overlapping resource intervals: {a}, {b}")
    partition = (summary["value_rate"] + summary["downstream_lost_value_rate"]
                 + summary["uncaptured_lost_value_rate"])
    if summary["total_value"] and abs(partition - 1.0) > 1e-8:
        raise AssertionError(f"disjoint outcome partition does not sum to one: {partition}")
    if abs(summary["capture_value_rate"] - summary["value_rate"]
           - summary["downstream_lost_value_rate"]) > 1e-8:
        raise AssertionError("capture value should equal completed plus downstream-lost value")
    if not (-model.EPS <= summary["value_rate"] <= 1 + model.EPS):
        raise AssertionError("value rate outside [0,1]")
    if summary["edge_utilization_0_h"] > 1 + model.EPS or summary["vehicle_utilization_0_h"] > 1 + model.EPS:
        raise AssertionError("fixed-window utilization exceeds one")
    deadlines = {r["task_id"]: r["deadline"] for r in rows}
    started = [e for e in logs if e["kind"] in ("capture_start", "edge_start")]
    for e in started:
        if e["time"] > deadlines[e["task_id"]] + model.EPS:
            raise AssertionError("operation started after deadline")


def audit_action_logs(tasks, vehicles, multiplier, rows, logs):
    """Independently recompute public plans and actual starts from raw event logs."""
    task_by_id = {t.task_id: t for t in tasks}
    row_by_id = {int(r["task_id"]): r for r in rows}
    positions = [(v.x, v.y) for v in vehicles]
    active_captures = [None] * len(vehicles)
    active_edge = None
    planned_routes = None
    planned_edge_order = None
    for e in logs:
        kind = e["kind"]
        if kind == "plan":
            now = e["time"]
            visible = set(e["visible_ids"])
            status = {int(j): st for j, st in e["status_before"].items()}
            routes = tuple(tuple(r) for r in e["routes"])
            if len(routes) != len(vehicles):
                raise AssertionError("plan route count differs from fleet size")
            route_ids = [j for route in routes for j in route]
            edge_order = tuple(e["edge_order"])
            if len(route_ids) != len(set(route_ids)) or len(edge_order) != len(set(edge_order)):
                raise AssertionError("plan contains duplicate task IDs")
            if not set(route_ids + list(edge_order)).issubset(visible):
                raise AssertionError("planner action contains an unpublished task")
            if any(task_by_id[j].release > now + model.EPS for j in visible):
                raise AssertionError("snapshot exposes a future task")

            active_map = e["active_vehicles"]
            report_release = {}
            for j, st in status.items():
                if st == "ready":
                    report_release[j] = now
                elif st == "uploading":
                    ready = row_by_id[j]["ready_time"]
                    if ready is None:
                        raise AssertionError("uploading report has no actual ready time")
                    report_release[j] = float(ready)
                elif st == "active_capture":
                    active = next((v for v in active_map.values() if v is not None and int(v["task_id"]) == j), None)
                    if active is None:
                        raise AssertionError("active capture absent from public vehicle state")
                    report_release[j] = float(active["finish"]) + task_by_id[j].upload

            captures = {}
            unique = set()
            for vid, route in enumerate(routes):
                v = vehicles[vid]
                active = active_map[str(vid)]
                if active is None:
                    cursor, loc = now, positions[vid]
                else:
                    active_id = int(active["task_id"])
                    if active_captures[vid] != active_id:
                        raise AssertionError("logged active capture does not match reconstructed vehicle state")
                    cursor = float(active["finish"])
                    loc = (task_by_id[active_id].x, task_by_id[active_id].y)
                for j in route:
                    if j not in visible or status.get(j) != "pending":
                        raise AssertionError("route uses a task that is not a visible pending task")
                    if j in unique:
                        raise AssertionError("task assigned to more than one vehicle")
                    unique.add(j)
                    t = task_by_id[j]
                    if t.sensor != v.sensor:
                        raise AssertionError("sensor-incompatible vehicle assignment")
                    leg = audit_travel_time(loc[0], loc[1], t.x, t.y, v.speed)
                    cursor += leg + t.service
                    if cursor + t.upload + t.work * multiplier > t.deadline + model.EPS:
                        raise AssertionError("route violates independent end-to-end deadline bound")
                    captures[j] = cursor
                    report_release[j] = cursor + t.upload
                    loc = (t.x, t.y)

            if not set(edge_order).issubset(report_release):
                raise AssertionError("edge plan uses a report that is neither ready nor predicted")
            edge_cursor = float(active_edge[2]) if active_edge is not None else now
            logged_times = e["edge_times"]
            for j in edge_order:
                start = max(edge_cursor, report_release[j])
                end = start + task_by_id[j].work * multiplier
                pair = logged_times.get(str(j), logged_times.get(j))
                if pair is None or abs(float(pair[0]) - start) > model.EPS or abs(float(pair[1]) - end) > model.EPS:
                    raise AssertionError("logged edge schedule differs from independent recomputation")
                if end > task_by_id[j].deadline + model.EPS:
                    raise AssertionError("edge plan includes a late completion")
                edge_cursor = end
            if abs(float(e["planned_value"]) - sum(task_by_id[j].value for j in edge_order)) > model.EPS:
                raise AssertionError("plan's objective value does not match its selected reports")
            planned_routes, planned_edge_order = routes, edge_order

        elif kind == "capture_start":
            vid, j = int(e["vehicle_id"]), int(e["task_id"])
            if active_captures[vid] is not None:
                raise AssertionError("vehicle started a second concurrent capture")
            if planned_routes is None or not planned_routes[vid] or planned_routes[vid][0] != j:
                raise AssertionError("actual vehicle start does not match the published plan head")
            t, v = task_by_id[j], vehicles[vid]
            leg = audit_travel_time(positions[vid][0], positions[vid][1], t.x, t.y, v.speed)
            if abs(float(e["travel"]) - leg) > model.EPS:
                raise AssertionError("actual travel time differs from independently reconstructed route")
            if abs(float(e["finish"]) - (e["time"] + leg + t.service)) > model.EPS:
                raise AssertionError("actual capture finish time is inconsistent")
            active_captures[vid] = j

        elif kind == "capture_complete":
            vid, j = int(e["vehicle_id"]), int(e["task_id"])
            if active_captures[vid] != j:
                raise AssertionError("capture completion does not match the occupied vehicle")
            positions[vid] = (task_by_id[j].x, task_by_id[j].y)
            active_captures[vid] = None

        elif kind == "edge_start":
            j = int(e["task_id"])
            if active_edge is not None:
                raise AssertionError("shared edge started overlapping jobs")
            if not planned_edge_order or planned_edge_order[0] != j:
                raise AssertionError("actual edge start does not match the published plan head")
            if abs(e["time"] + task_by_id[j].work * multiplier - e["finish"]) > model.EPS:
                raise AssertionError("actual edge finish time is inconsistent")
            active_edge = (j, e["time"], e["finish"])

        elif kind == "edge_complete":
            j = int(e["task_id"])
            if active_edge is None or active_edge[0] != j:
                raise AssertionError("edge completion does not match the occupied server")
            if e["success"] and active_edge[2] > task_by_id[j].deadline + model.EPS:
                raise AssertionError("late edge completion counted as success")
            active_edge = None
