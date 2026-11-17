"""Current deterministic acquisition and edge-processing simulator."""
from __future__ import annotations

from dataclasses import dataclass, field, replace, asdict
from pathlib import Path
import copy
import hashlib
import importlib.util
import itertools
import json
import math
import random
import sys
import time

ROOT = Path(__file__).resolve().parent
LEGACY_FILE = ROOT / "routing.py"
spec = importlib.util.spec_from_file_location("runtime_routes", LEGACY_FILE)
legacy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = legacy
spec.loader.exec_module(legacy)
EPS = 1e-8
H = 50.0


@dataclass(frozen=True)
class Profile:
    name: str
    arrival_rate: float = 1.0
    edge_multiplier: float = 1.5
    slack: tuple[float, ...] = (16.0, 24.0)


PROFILES = (
    Profile("ample_edge", edge_multiplier=0.1),
    Profile("coupled_primary"),
    Profile("edge_dominant", edge_multiplier=2.5),
    Profile("tighter_deadlines", slack=(8.0, 12.0)),
    Profile("low_load", arrival_rate=0.45),
)


@dataclass(frozen=True)
class Task:
    task_id: int
    release: float
    x: int
    y: int
    deadline: float
    value: float
    sensor: int
    service: float
    upload: float = 0.2
    work: float = 1.0


@dataclass
class Snapshot:
    now: float
    tasks: dict[int, Task]
    status: dict[int, str]
    vehicles: list
    ready_times: dict[int, float]
    edge_active: tuple[int, float, float] | None
    multiplier: float
    event_index: int = 0


@dataclass
class Plan:
    routes: tuple[tuple[int, ...], ...]
    edge_order: tuple[int, ...]
    edge_times: dict[int, tuple[float, float]]
    capture_times: dict[int, float]
    value: float
    count: int
    wait: float
    travel: float
    rule: str
    repairs: int = 0

    @property
    def score(self):
        return (self.value, self.count, -self.wait, -self.travel)

    @property
    def key(self):
        return self.routes, self.edge_order


def generate(seed: int, profile: Profile, horizon=H, n_vehicles=8):
    old_profile = legacy.Profile(profile.name, profile.arrival_rate, (2.0, 4.0, 6.0), profile.slack, "v05")
    old_tasks, vehicles = legacy.generate_instance(seed, old_profile, horizon, n_vehicles)
    rng = random.Random(seed + 607)
    tasks = [Task(**asdict(t), upload=0.2, work=rng.choice((0.5, 1.0, 1.5))) for t in old_tasks]
    return tasks, vehicles


def signature(tasks, vehicles):
    doc = {"tasks": [asdict(t) for t in tasks], "vehicles": [asdict(v) for v in vehicles]}
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def route_base(s: Snapshot, vid: int):
    v = s.vehicles[vid]
    if v.active is None:
        return s.now, (v.x, v.y)
    t = s.tasks[v.active.task_id]
    return v.active.finish, (t.x, t.y)


def route_predictions(s: Snapshot, routes):
    """Independent minimum e2e check, continuity and unique pending jobs."""
    seen, captures, travel = set(), {}, 0.0
    if len(routes) != len(s.vehicles):
        return None
    for vid, route in enumerate(routes):
        cursor, loc = route_base(s, vid)
        v = s.vehicles[vid]
        for j in route:
            if j in seen or s.status.get(j) != "pending":
                return None
            t = s.tasks[j]
            if t.sensor != v.sensor:
                return None
            leg = legacy.travel_time(*loc, t.x, t.y, v.speed)
            cursor += leg + t.service
            if cursor + t.upload + t.work * s.multiplier > t.deadline + EPS:
                return None
            captures[j] = cursor
            travel += leg
            seen.add(j)
            loc = (t.x, t.y)
    return captures, travel


def report_releases(s: Snapshot, captures):
    reports = {}
    for j, status in s.status.items():
        if status in ("ready", "uploading"):
            reports[j] = max(s.now, s.ready_times[j])
    for v in s.vehicles:
        if v.active and s.status[v.active.task_id] != "expired":
            j = v.active.task_id
            reports[j] = v.active.finish + s.tasks[j].upload
    reports.update({j: c + s.tasks[j].upload for j, c in captures.items()})
    return reports


def edge_cursor(s):
    return s.now if s.edge_active is None else s.edge_active[2]


def edge_schedule(s: Snapshot, releases: dict[int, float], rule: str, required=frozenset()):
    """Public common scheduling module. Explicit future reservation allowed."""
    if rule == "exact":
        return exact_edge(s, releases, required)
    cursor = edge_cursor(s)
    pending = set(releases)
    times, order = {}, []
    while pending:
        if rule == "reserved_edf":
            j = min(pending, key=lambda k: (s.tasks[k].deadline, releases[k], k))
        else:
            ready = [j for j in pending if releases[j] <= cursor + EPS]
            if not ready:
                cursor = max(cursor, min(releases[j] for j in pending))
                ready = [j for j in pending if releases[j] <= cursor + EPS]
            if rule == "edf":
                j = min(ready, key=lambda k: (s.tasks[k].deadline, releases[k], k))
            elif rule == "density":
                j = min(ready, key=lambda k: (-s.tasks[k].value / (s.tasks[k].work * s.multiplier), s.tasks[k].deadline, k))
            else:
                raise ValueError(rule)
        pending.remove(j)
        start = max(cursor, releases[j])
        end = start + s.tasks[j].work * s.multiplier
        if end > s.tasks[j].deadline + EPS:
            continue
        times[j] = (start, end)
        order.append(j)
        cursor = end
    if not set(required).issubset(times):
        return None
    return tuple(order), times


def exact_edge(s: Snapshot, releases, required=frozenset(), max_jobs=10):
    """Exact value/count/min-final-time subset DP for FIXED report releases.

    It is not claimed to minimize sum waiting or solve vehicle routing.
    For one completed subset, smallest cursor dominates all later schedules.
    """
    ids = sorted(releases)
    if len(ids) > max_jobs:
        raise ValueError("exact edge diagnostic cap exceeded")
    states = {0: (edge_cursor(s), ())}
    for mask in range(1 << len(ids)):
        if mask not in states:
            continue
        cursor, path = states[mask]
        for k, j in enumerate(ids):
            if mask & (1 << k):
                continue
            end = max(cursor, releases[j]) + s.tasks[j].work * s.multiplier
            if end > s.tasks[j].deadline + EPS:
                continue
            new_mask = mask | (1 << k)
            new = (end, path + (j,))
            if new_mask not in states or new < states[new_mask]:
                states[new_mask] = new
    valid = [(sum(s.tasks[j].value for j in path), len(path), -end, path)
             for end, path in states.values() if set(required).issubset(path)]
    if not valid:
        return None
    best = min(valid, key=lambda x: (-x[0], -x[1], -x[2], x[3]))
    order = best[3]
    cursor, times = edge_cursor(s), {}
    for j in order:
        start = max(cursor, releases[j])
        cursor = start + s.tasks[j].work * s.multiplier
        times[j] = (start, cursor)
    return order, times


def evaluate(s: Snapshot, routes, rules=("edf", "reserved_edf", "density"), required=None,
             eligible_reports=None):
    pred = route_predictions(s, routes)
    if pred is None:
        return None
    captures, travel = pred
    releases = report_releases(s, captures)
    required = set(captures) if required is None else set(required)
    if eligible_reports is not None:
        eligible = set(eligible_reports) | set(captures)
        releases = {j: r for j, r in releases.items() if j in eligible}
    options = []
    for rule in rules:
        result = edge_schedule(s, releases, rule, required)
        if result is None:
            continue
        order, times = result
        p = Plan(tuple(tuple(r) for r in routes), order, times, captures,
                 sum(s.tasks[j].value for j in order), len(order),
                 sum(end - s.tasks[j].release for j, (_, end) in times.items()), travel, rule)
        options.append(p)
    return min(options, key=lambda p: tuple(-x for x in p.score) + (p.key,)) if options else None


def empty_routes(s):
    return tuple(() for _ in s.vehicles)


def public_legacy_inputs(s):
    tasks = {j: legacy.Task(t.task_id, t.release, t.x, t.y,
                           t.deadline - t.upload - t.work * s.multiplier,
                           t.value, t.sensor, t.service) for j, t in s.tasks.items()}
    return copy.deepcopy(s.vehicles), tasks


def vehicle_plan(s, method):
    vehicles, tasks = public_legacy_inputs(s)
    pending = {j for j, st in s.status.items() if st == "pending"}
    if method in ("vehicle_repack_edf", "edge_adapter"):
        for v in vehicles:
            v.queue.clear()
    else:
        pending.difference_update(j for v in vehicles for j in v.queue)
    if method == "edge_adapter":
        legacy.edge_coalition_plan(pending, vehicles, s.now, tasks, s.event_index)
    else:
        legacy.make_plan("density_insertion", pending, vehicles, s.now, tasks)
    routes = tuple(tuple(v.queue) for v in vehicles)
    # Separated strategy can lose reports downstream; physical minimum check
    # still applies. Edge adapter gets best-of-three, without weakening routing.
    rules = ("edf", "reserved_edf", "density") if method == "edge_adapter" else ("edf",)
    plan = evaluate(s, routes, rules, required=set())
    assert plan is not None
    return plan


def greedy_e2e(s, repack=True, rule_set=("edf", "reserved_edf", "density"), initial=None):
    routes = [list(r) for r in (initial if initial is not None else
              empty_routes(s) if repack else tuple(tuple(v.queue) for v in s.vehicles))]
    # Previously chosen soft queues can be invalid after an intervening policy
    # event; explicit repair returns unservable soft jobs to pending.
    while True:
        base = evaluate(s, routes, rule_set)
        if base is not None:
            break
        best_partial = evaluate(s, routes, rule_set, required=set())
        served = set(best_partial.edge_order) if best_partial else set()
        removed = False
        for r in routes:
            for j in r[:]:
                if j not in served:
                    r.remove(j)
                    removed = True
        if not removed:
            routes = [list(r) for r in empty_routes(s)]
    remaining = {j for j, st in s.status.items() if st == "pending"} - set(base.capture_times)
    while remaining:
        choices = []
        protected = set(base.edge_order)
        for j in sorted(remaining):
            t = s.tasks[j]
            for vid, vehicle in enumerate(s.vehicles):
                if vehicle.sensor != t.sensor:
                    continue
                for pos in range(len(routes[vid]) + 1):
                    new_routes = [r[:] for r in routes]
                    new_routes[vid].insert(pos, j)
                    trial = evaluate(s, new_routes, rule_set, required=protected | {j})
                    if trial is None:
                        continue
                    marginal = max(t.service, trial.travel - base.travel + t.service) + t.work * s.multiplier
                    choices.append((-(t.value / marginal), t.deadline,
                                    tuple(-x for x in trial.score), j, vid, pos, trial))
        if not choices:
            break
        chosen = min(choices, key=lambda x: x[:-1])
        j, vid, pos, base = chosen[3:]
        routes[vid].insert(pos, j)
        remaining.remove(j)
    return base


def joint_resource_density(s, rule_set=("edf", "reserved_edf", "density")):
    """Transparent joint greedy over report jobs and new capture jobs.

    Each step inserts the feasible candidate with largest marginal value per
    incremental vehicle-plus-edge time. It may replace one previously selected
    already-captured report, so an arbitrary first report choice cannot block a
    more valuable new end-to-end task. All current soft capture jobs are rebuilt.
    """
    routes = [list(r) for r in empty_routes(s)]
    selected_reports = set()
    base = evaluate(s, routes, rule_set, required=set(), eligible_reports=set())
    remaining = {j for j, st in s.status.items() if st == "pending"}
    existing_reports = set(report_releases(s, {}))

    while True:
        choices = []
        route_jobs = {j for route in routes for j in route}
        drop_options = [None, *sorted(selected_reports)]

        # A report that has already consumed vehicle time is an edge-only job.
        for drop in drop_options:
            retained = selected_reports - ({drop} if drop is not None else set())
            for j in sorted(existing_reports - selected_reports):
                req = retained | route_jobs | {j}
                trial = evaluate(s, routes, rule_set, required=req,
                                 eligible_reports=req)
                if trial is None:
                    continue
                gain = trial.value - base.value
                if gain <= EPS:
                    continue
                resource = max(EPS, s.tasks[j].work * s.multiplier)
                choices.append((-gain / resource, -gain, s.tasks[j].deadline,
                                j, -1 if drop is None else drop, -1, -1,
                                trial, retained | {j}))

        # A new task consumes vehicle time and a future edge-processing slot.
        for j in sorted(remaining):
            t = s.tasks[j]
            for vid, vehicle in enumerate(s.vehicles):
                if vehicle.sensor != t.sensor:
                    continue
                for pos in range(len(routes[vid]) + 1):
                    new_routes = [r[:] for r in routes]
                    new_routes[vid].insert(pos, j)
                    new_jobs = route_jobs | {j}
                    for drop in drop_options:
                        retained = selected_reports - ({drop} if drop is not None else set())
                        req = retained | new_jobs
                        trial = evaluate(s, new_routes, rule_set, required=req,
                                         eligible_reports=req)
                        if trial is None:
                            continue
                        gain = trial.value - base.value
                        if gain <= EPS:
                            continue
                        added_travel = max(0.0, trial.travel - base.travel)
                        resource = max(EPS, added_travel + t.service + t.work * s.multiplier)
                        choices.append((-gain / resource, -gain, t.deadline,
                                        j, -1 if drop is None else drop, vid, pos,
                                        trial, retained))
        if not choices:
            break
        chosen = min(choices, key=lambda x: x[:-2])
        j, drop, vid, pos, trial, retained = chosen[3:]
        if vid >= 0:
            routes[vid].insert(pos, j)
            remaining.remove(j)
        base = trial
        selected_reports = retained
    return base


def exact_joint(s, max_pending=4, max_reports=8):
    """Enumerate ordered vehicle partitions + exact edge subset DP.

    Exact for current snapshot value only; no future arrivals. Tiny diagnostic.
    """
    ids = sorted(j for j, st in s.status.items() if st == "pending")
    if len(ids) > max_pending:
        raise ValueError("exact joint pending cap exceeded")
    snapshot_reports = set(report_releases(s, {})) | set(ids)
    if len(snapshot_reports) > max_reports:
        raise ValueError("exact joint report cap exceeded; snapshot is outside the declared subproblem")
    best = evaluate(s, empty_routes(s), ("exact",))
    seen = set()
    for size in range(len(ids) + 1):
        for subset in itertools.combinations(ids, size):
            eligible = [[i for i, v in enumerate(s.vehicles) if v.sensor == s.tasks[j].sensor] for j in subset]
            for assignment in itertools.product(*eligible):
                for order in itertools.permutations(range(size)):
                    routes = [[] for _ in s.vehicles]
                    for k in order:
                        routes[assignment[k]].append(subset[k])
                    key = tuple(tuple(r) for r in routes)
                    if key in seen:
                        continue
                    seen.add(key)
                    pred = route_predictions(s, routes)
                    if pred is None:
                        continue
                    trial = evaluate(s, routes, ("exact",))
                    if trial and (trial.value > best.value + EPS or
                                  (abs(trial.value - best.value) < EPS and
                                   (trial.count, -trial.wait, -trial.travel) > (best.count, -best.wait, -best.travel))):
                        best = trial
    return best, len(seen)


METHODS = ("vehicle_insertion_edf", "vehicle_repack_edf", "e2e_insertion", "e2e_repack",
           "e2e_resource_density", "edge_adapter")


def make_plan(s, method):
    if method in ("vehicle_insertion_edf", "vehicle_repack_edf", "edge_adapter"):
        return vehicle_plan(s, method)
    if method in ("e2e_insertion", "e2e_repack"):
        return greedy_e2e(s, repack=method == "e2e_repack")
    if method == "e2e_resource_density":
        return joint_resource_density(s)
    raise ValueError(method)


def copy_vehicles_for_snapshot(vehicles):
    """Structural copy of vehicle state for a read-only planning snapshot.

    Equivalent to deepcopy for this model: Env/NetworkEnv rebind vehicle
    fields (queue, active, x, y) and never mutate ActiveJob or queue entries
    in place, so copying each mutable leaf once is sufficient and much
    cheaper than deepcopy's recursive memo machinery.
    """
    out = []
    for v in vehicles:
        c = copy.copy(v)
        c.active = None if v.active is None else copy.copy(v.active)
        c.queue = list(v.queue)
        out.append(c)
    return out


class Env:
    def __init__(self, tasks, vehicles, multiplier):
        self._stream = sorted(tasks, key=lambda t: (t.release, t.task_id))
        self.vehicles = copy.deepcopy(vehicles)
        self.multiplier = multiplier
        self.known, self.status, self.ready_times = {}, {}, {}
        self.edge_active = None
        self.arrival_index = 0
        self.now = 0.0
        self.event_index = 0
        self.logs = []
        self.records = {}
        self.plan = None
        self.planning_seconds = 0.0
        self._settle()

    def log(self, kind, **data):
        self.logs.append({"event_index": self.event_index, "time": self.now, "kind": kind, **data})

    def snapshot(self):
        return Snapshot(self.now, self.known.copy(), self.status.copy(),
                        copy_vehicles_for_snapshot(self.vehicles),
                        self.ready_times.copy(), self.edge_active, self.multiplier, self.event_index)

    def _settle(self):
        self.event_index += 1
        if self.edge_active and self.edge_active[2] <= self.now + EPS:
            j, start, end = self.edge_active
            t = self.known[j]
            success = end <= t.deadline + EPS and self.status[j] != "expired"
            if success:
                self.status[j] = "completed"
            self.records[j]["edge_finish"] = end
            self.log("edge_complete", task_id=j, success=success)
            self.edge_active = None
        for vid, v in enumerate(self.vehicles):
            if v.active and v.active.finish <= self.now + EPS:
                j, end = v.active.task_id, v.active.finish
                t = self.known[j]
                v.x, v.y, v.active = t.x, t.y, None
                self.records[j]["capture_finish"] = end
                if self.status[j] != "expired":
                    self.status[j] = "uploading"
                    self.ready_times[j] = end + t.upload
                self.log("capture_complete", task_id=j, vehicle_id=vid)
        while self.arrival_index < len(self._stream) and self._stream[self.arrival_index].release <= self.now + EPS:
            t = self._stream[self.arrival_index]
            self.known[t.task_id] = t
            self.status[t.task_id] = "pending"
            self.records[t.task_id] = {**asdict(t), "status": "pending", "vehicle_id": None,
                                      "capture_start": None, "capture_finish": None,
                                      "travel_time": 0.0, "edge_start": None, "edge_finish": None}
            self.log("arrival", task_id=t.task_id)
            self.arrival_index += 1
        for j, st in list(self.status.items()):
            if st == "uploading" and self.ready_times[j] <= self.now + EPS:
                self.status[j] = "ready"
                self.log("ready", task_id=j)
        for j, st in list(self.status.items()):
            if st not in ("completed", "expired") and self.known[j].deadline <= self.now + EPS:
                self.status[j] = "expired"
                self.log("expiry", task_id=j, previous_status=st)
        for v in self.vehicles:
            v.queue = [j for j in v.queue if self.status.get(j) == "pending"]

    @property
    def done(self):
        return (self.arrival_index == len(self._stream)
                and all(st in ("completed", "expired") for st in self.status.values())
                and self.edge_active is None and all(v.active is None for v in self.vehicles))

    def apply(self, p: Plan):
        s = self.snapshot()
        pred = route_predictions(s, p.routes)
        if pred is None:
            raise AssertionError("illegal vehicle plan")
        releases = report_releases(s, pred[0])
        assert len(p.edge_order) == len(set(p.edge_order))
        cursor = edge_cursor(s)
        for j in p.edge_order:
            assert j in releases, "unknown/non-report task in edge plan"
            start = max(cursor, releases[j])
            end = start + s.tasks[j].work * s.multiplier
            assert end <= s.tasks[j].deadline + EPS
            assert abs(start - p.edge_times[j][0]) < EPS and abs(end - p.edge_times[j][1]) < EPS
            cursor = end
        self.plan = p
        self.log("plan", routes=p.routes, edge_order=p.edge_order,
                 edge_times={str(j): t for j, t in p.edge_times.items()},
                 visible_ids=sorted(s.tasks), status_before=s.status,
                 active_vehicles={str(i): None if v.active is None else asdict(v.active) for i, v in enumerate(s.vehicles)},
                 edge_active=s.edge_active, planned_value=p.value, rule=p.rule)
        for vid, v in enumerate(self.vehicles):
            v.queue = list(p.routes[vid])
            if v.active or not v.queue:
                continue
            j = v.queue.pop(0)
            t = self.known[j]
            assert self.status[j] == "pending"
            leg = legacy.travel_time(v.x, v.y, t.x, t.y, v.speed)
            finish = self.now + leg + t.service
            assert finish + t.upload + t.work * self.multiplier <= t.deadline + EPS
            v.active = legacy.ActiveJob(j, self.now, finish, leg)
            self.status[j] = "active_capture"
            self.records[j].update(vehicle_id=vid, capture_start=self.now, travel_time=leg)
            self.log("capture_start", task_id=j, vehicle_id=vid, finish=finish, travel=leg)
        if self.edge_active is None and p.edge_order:
            j = p.edge_order[0]
            if self.status[j] == "ready":
                end = self.now + self.known[j].work * self.multiplier
                assert end <= self.known[j].deadline + EPS
                assert abs(p.edge_times[j][0] - self.now) < EPS
                self.edge_active = (j, self.now, end)
                self.status[j] = "active_edge"
                self.records[j]["edge_start"] = self.now
                self.log("edge_start", task_id=j, finish=end)
            else:
                self.log("edge_wait", task_id=j, ready_expected=releases[j])

    def advance(self):
        future = []
        if self.arrival_index < len(self._stream):
            future.append(self._stream[self.arrival_index].release)
        future.extend(v.active.finish for v in self.vehicles if v.active)
        if self.edge_active:
            future.append(self.edge_active[2])
        future.extend(self.ready_times[j] for j, st in self.status.items() if st == "uploading")
        future.extend(self.known[j].deadline for j, st in self.status.items() if st not in ("completed", "expired"))
        future = [t for t in future if t > self.now + EPS]
        if not future:
            if not self.done:
                raise RuntimeError("nonterminal state without event")
            return
        self.now = min(future)
        self._settle()

    def results(self):
        rows = []
        for j in sorted(self.records):
            r = dict(self.records[j])
            r["status"] = self.status[j]
            r["ready_time"] = self.ready_times.get(j)
            rows.append(r)
        total = sum(r["value"] for r in rows)
        completed = [r for r in rows if r["status"] == "completed"]
        captured = [r for r in rows if r["capture_finish"] is not None]
        expired_captured = [r for r in captured if r["status"] != "completed"]
        rv = lambda rr: sum(r["value"] for r in rr) / total if total else 0.0
        capture_busy_before_h = sum(
            max(0.0, min(r["capture_finish"], H) - max(r["capture_start"], 0.0))
            for r in rows if r["capture_start"] is not None and r["capture_finish"] is not None)
        edge_busy_before_h = sum(
            max(0.0, min(r["edge_finish"], H) - max(r["edge_start"], 0.0))
            for r in rows if r["edge_start"] is not None and r["edge_finish"] is not None)
        capture_busy_after_h = sum(
            max(0.0, r["capture_finish"] - max(r["capture_start"], H))
            for r in rows if r["capture_start"] is not None and r["capture_finish"] is not None)
        edge_busy_after_h = sum(
            max(0.0, r["edge_finish"] - max(r["edge_start"], H))
            for r in rows if r["edge_start"] is not None and r["edge_finish"] is not None)
        completed_latency = [r["edge_finish"] - r["release"] for r in completed]
        processed_latency = [r["edge_finish"] - r["release"] for r in rows if r["edge_finish"] is not None]
        summary = {"n_tasks": len(rows), "total_value": total,
                   "completed_value": sum(r["value"] for r in completed), "value_rate": rv(completed),
                   "count_rate": len(completed) / len(rows) if rows else 0.0,
                   "capture_value_rate": rv(captured), "downstream_lost_value_rate": rv(expired_captured),
                   "uncaptured_lost_value_rate": 1.0 - rv(captured) if total else 0.0,
                   "travel_time": sum(r["travel_time"] for r in rows),
                   "capture_time": sum(r["travel_time"] + r["service"] for r in rows if r["capture_start"] is not None),
                   "edge_busy_time": sum(r["work"] * self.multiplier for r in rows if r["edge_start"] is not None),
                   "edge_wait_time": sum(r["edge_start"] - r["ready_time"] for r in rows if r["edge_start"] is not None),
                   "capture_busy_before_h": capture_busy_before_h,
                   "edge_busy_before_h": edge_busy_before_h,
                   "vehicle_utilization_0_h": capture_busy_before_h / (len(self.vehicles) * H),
                   "edge_utilization_0_h": edge_busy_before_h / H,
                   "capture_busy_after_h": capture_busy_after_h,
                   "edge_busy_after_h": edge_busy_after_h,
                   "completed_latency_n": len(completed_latency),
                   "completed_latency_mean": sum(completed_latency) / len(completed_latency) if completed_latency else 0.0,
                   "processed_latency_n": len(processed_latency),
                   "processed_latency_mean": sum(processed_latency) / len(processed_latency) if processed_latency else 0.0,
                   "wasted_capture_time": sum(r["travel_time"] + r["service"] for r in expired_captured),
                   "end_time": self.now, "events": self.event_index,
                   "planning_ms": self.planning_seconds * 1000}
        return summary, rows, self.logs


def run_episode(seed, profile, method, policy=None):
    tasks, vehicles = generate(seed, profile)
    stream_hash = signature(tasks, vehicles)
    env = Env(tasks, vehicles, profile.edge_multiplier)
    while not env.done:
        s = env.snapshot()
        started = time.perf_counter()
        p = policy(s) if policy else make_plan(s, method)
        env.planning_seconds += time.perf_counter() - started
        env.apply(p)
        env.advance()
        if env.event_index > 10000:
            raise AssertionError("event loop did not terminate")
    summary, rows, logs = env.results()
    summary.update(seed=seed, profile=profile.name, method=method, stream_hash=stream_hash)
    for r in rows:
        r.update(seed=seed, profile=profile.name, method=method)
    for e in logs:
        e.update(seed=seed, profile=profile.name, method=method)
    return summary, rows, logs
