"""Current simulator routing primitives and required comparison operators."""


from __future__ import annotations


import argparse


import csv


import hashlib


import json


import math


import random


import statistics


import time


from dataclasses import asdict, dataclass, field


from pathlib import Path


from typing import Iterable


EPS = 1e-9


@dataclass(frozen=True)
class Profile:
    name: str
    arrival_rate: float
    service_values: tuple[float, ...]
    slack_values: tuple[float, ...]
    purpose: str


PROFILES = (
    Profile("low_load_negative_control", 0.45, (0.5, 1.0), (16.0, 24.0), "low-load negative control"),
    Profile("short_service_wide_deadline", 1.0, (0.5, 1.0), (16.0, 24.0), "short service and wide deadlines"),
    Profile("long_service_wide_deadline", 1.0, (2.0, 4.0, 6.0), (16.0, 24.0), "persistent occupation isolated"),
    Profile("short_service_tight_deadline", 1.0, (0.5, 1.0), (4.0, 8.0), "deadline pressure isolated"),
    Profile("long_service_tight_deadline", 1.0, (2.0, 4.0, 6.0), (4.0, 8.0), "occupation and deadline interaction"),
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


@dataclass
class ActiveJob:
    task_id: int
    started: float
    finish: float
    travel: float


@dataclass
class Vehicle:
    vehicle_id: int
    x: int
    y: int
    sensor: int
    speed: float
    active: ActiveJob | None = None
    queue: list[int] = field(default_factory=list)


def generate_instance(seed: int, profile: Profile, horizon: float = 50.0, n_vehicles: int = 8):
    """Generate common exogenous streams; all methods see exactly the same instance."""
    arrival_rng = random.Random(seed + 101)
    attr_rng = random.Random(seed + 211)
    service_rng = random.Random(seed + 307)
    slack_rng = random.Random(seed + 401)
    vehicle_rng = random.Random(seed + 503)

    # A unit-rate Poisson process is time-scaled by lambda, preserving common
    # arrival quantiles when workload profiles change.
    arrivals: list[float] = []
    t = 0.0
    while True:
        t += arrival_rng.expovariate(1.0) / profile.arrival_rate
        if t >= horizon:
            break
        arrivals.append(t)

    tasks: list[Task] = []
    for task_id, release in enumerate(arrivals):
        tasks.append(
            Task(
                task_id=task_id,
                release=release,
                x=attr_rng.randrange(10),
                y=attr_rng.randrange(10),
                deadline=release + slack_rng.choice(profile.slack_values),
                value=float(attr_rng.randint(1, 10)),
                sensor=attr_rng.randrange(2),
                service=service_rng.choice(profile.service_values),
            )
        )

    sensor_types = [0] * (n_vehicles // 2) + [1] * (n_vehicles - n_vehicles // 2)
    vehicle_rng.shuffle(sensor_types)
    vehicles = [
        Vehicle(
            vehicle_id=i,
            x=vehicle_rng.randrange(10),
            y=vehicle_rng.randrange(10),
            sensor=sensor_types[i],
            speed=vehicle_rng.uniform(1.5, 2.5),
        )
        for i in range(n_vehicles)
    ]
    return tasks, vehicles


def travel_time(x1: int, y1: int, x2: int, y2: int, speed: float) -> float:
    return (abs(x1 - x2) + abs(y1 - y2)) / speed


def route_base(vehicle: Vehicle, now: float, tasks: dict[int, Task]):
    if vehicle.active is None:
        return now, (vehicle.x, vehicle.y)
    active_task = tasks[vehicle.active.task_id]
    return vehicle.active.finish, (active_task.x, active_task.y)


def evaluate_route(vehicle: Vehicle, route: list[int], now: float, tasks: dict[int, Task]):
    """Return feasibility, task completion times, route travel, and route end."""
    cursor_t, loc = route_base(vehicle, now, tasks)
    base_t = cursor_t
    completion: dict[int, float] = {}
    total_travel = 0.0
    for task_id in route:
        task = tasks[task_id]
        if task.sensor != vehicle.sensor:
            return False, {}, math.inf, math.inf
        leg = travel_time(loc[0], loc[1], task.x, task.y, vehicle.speed)
        total_travel += leg
        cursor_t += leg + task.service
        completion[task_id] = cursor_t
        if cursor_t > task.deadline + EPS:
            return False, completion, total_travel, cursor_t
        loc = (task.x, task.y)
    return True, completion, total_travel, cursor_t


def route_duration(vehicle: Vehicle, route: list[int], now: float, tasks: dict[int, Task]) -> float:
    feasible, _, _, end = evaluate_route(vehicle, route, now, tasks)
    if not feasible:
        return math.inf
    base_t, _ = route_base(vehicle, now, tasks)
    return end - base_t


def candidate_insertions(pending: Iterable[int], vehicles, now, tasks, insertion: bool):
    candidates = []
    for task_id in pending:
        task = tasks[task_id]
        for vehicle_id, vehicle in enumerate(vehicles):
            if task.sensor != vehicle.sensor:
                continue
            original = vehicle.queue
            old_duration = route_duration(vehicle, original, now, tasks)
            positions = range(len(original) + 1) if insertion else (len(original),)
            for pos in positions:
                route = original[:pos] + [task_id] + original[pos:]
                feasible, _, _, _ = evaluate_route(vehicle, route, now, tasks)
                if not feasible:
                    continue
                new_duration = route_duration(vehicle, route, now, tasks)
                marginal = max(task.service, new_duration - old_duration, 0.05)
                ratio = task.value / marginal
                candidates.append((ratio, task.deadline, task.release, task_id, vehicle_id, pos, marginal))
    # Stable tie-breaking is part of the frozen policy.
    candidates.sort(key=lambda c: (-c[0], c[1], c[2], c[3], c[4], c[5]))
    return candidates


def make_plan(method: str, pending: set[int], vehicles: list[Vehicle], now: float, tasks: dict[int, Task]):
    if method == "myopic_density":
        # A strictly one-step dispatcher: assigns at most one job to each idle
        # vehicle, and never pre-books jobs for currently busy vehicles.
        candidates = []
        for task_id in pending:
            task = tasks[task_id]
            for vid, vehicle in enumerate(vehicles):
                if vehicle.active is not None or vehicle.queue or task.sensor != vehicle.sensor:
                    continue
                leg = travel_time(vehicle.x, vehicle.y, task.x, task.y, vehicle.speed)
                finish = now + leg + task.service
                if finish <= task.deadline + EPS:
                    candidates.append((-(task.value / max(leg + task.service, 0.05)), task.deadline, task_id, vid))
        candidates.sort()
        assigned_vehicles = set()
        for _, _, task_id, vid in candidates:
            if task_id in pending and vid not in assigned_vehicles:
                vehicles[vid].queue.append(task_id)
                pending.remove(task_id)
                assigned_vehicles.add(vid)
        return

    if method == "edf_append":
        # Persistent deadline-order list scheduling; only append to each queue.
        for task_id in sorted(list(pending), key=lambda j: (tasks[j].deadline, tasks[j].release, j)):
            options = []
            for vid, vehicle in enumerate(vehicles):
                if tasks[task_id].sensor != vehicle.sensor:
                    continue
                route = vehicle.queue + [task_id]
                feasible, completion, _, _ = evaluate_route(vehicle, route, now, tasks)
                if feasible:
                    options.append((completion[task_id], route_duration(vehicle, route, now, tasks), vid))
            if options:
                _, _, vid = min(options)
                vehicles[vid].queue.append(task_id)
                pending.remove(task_id)
        return

    if method not in ("density_append", "density_insertion"):
        raise ValueError(f"unknown method: {method}")

    insertion = method == "density_insertion"
    while pending:
        candidates = candidate_insertions(pending, vehicles, now, tasks, insertion)
        if not candidates:
            break
        _, _, _, task_id, vid, pos, _ = candidates[0]
        vehicles[vid].queue.insert(pos, task_id)
        pending.remove(task_id)


def exact_route_options(vehicle: Vehicle, job_ids: list[int], now: float, tasks: dict[int, Task]):
    """Return a feasible low-wait route for every feasible job subset by DP."""
    ids = sorted(job_ids)
    n = len(ids)
    if n == 0:
        base_t, _ = route_base(vehicle, now, tasks)
        return {0: (0.0, base_t, ())}
    size = 1 << n
    # Each label is (last completion time, accumulated wait-to-service, route).
    # Keep a Pareto frontier because a lower final time and lower total wait do
    # not always come from the same task order.
    labels = [[[] for _ in range(n)] for _ in range(size)]
    base_t, loc = route_base(vehicle, now, tasks)

    def add_label(frontier, candidate):
        c_finish, c_wait, c_route = candidate
        for old_finish, old_wait, old_route in frontier:
            weakly_dominates = old_finish <= c_finish + EPS and old_wait <= c_wait + EPS
            strictly_dominates = old_finish < c_finish - EPS or old_wait < c_wait - EPS
            if weakly_dominates and (strictly_dominates or old_route <= c_route):
                return
        kept = []
        for old_finish, old_wait, old_route in frontier:
            weakly_dominated = c_finish <= old_finish + EPS and c_wait <= old_wait + EPS
            strictly_better = c_finish < old_finish - EPS or c_wait < old_wait - EPS
            if weakly_dominated and (strictly_better or c_route < old_route):
                continue
            kept.append((old_finish, old_wait, old_route))
        kept.append(candidate)
        kept.sort(key=lambda row: (row[0], row[1], row[2]))
        frontier[:] = kept

    for i, task_id in enumerate(ids):
        task = tasks[task_id]
        leg = travel_time(loc[0], loc[1], task.x, task.y, vehicle.speed)
        service_start = base_t + leg
        finish = base_t + leg + task.service
        if finish <= task.deadline + EPS:
            wait = max(0.0, service_start - task.release)
            add_label(labels[1 << i][i], (finish, wait, (task_id,)))

    route_for_mask: dict[int, tuple[float, float, tuple[int, ...]]] = {0: (0.0, base_t, ())}
    for mask in range(1, size):
        for last in range(n):
            last_task = tasks[ids[last]]
            for finish, wait_total, route in list(labels[mask][last]):
                candidate_for_mask = (wait_total, finish, route)
                current = route_for_mask.get(mask)
                if current is None or candidate_for_mask < current:
                    route_for_mask[mask] = candidate_for_mask
                for nxt in range(n):
                    bit = 1 << nxt
                    if mask & bit:
                        continue
                    next_task = tasks[ids[nxt]]
                    leg = travel_time(last_task.x, last_task.y, next_task.x, next_task.y, vehicle.speed)
                    service_start = finish + leg
                    next_finish = service_start + next_task.service
                    if next_finish > next_task.deadline + EPS:
                        continue
                    next_wait = wait_total + max(0.0, service_start - next_task.release)
                    new_mask = mask | bit
                    new_route = route + (ids[nxt],)
                    add_label(labels[new_mask][nxt], (next_finish, next_wait, new_route))
    return route_for_mask


def edge_route_metrics(route: list[int], vehicle: Vehicle, now: float, tasks: dict[int, Task]):
    """Deadline-completion feasibility and total response wait for one route."""
    cursor, loc = route_base(vehicle, now, tasks)
    total_wait = 0.0
    starts = {}
    for task_id in route:
        task = tasks[task_id]
        leg = travel_time(loc[0], loc[1], task.x, task.y, vehicle.speed)
        service_start = max(cursor + leg, task.release)
        finish = service_start + task.service
        if finish > task.deadline + EPS:
            return False, math.inf, {}
        starts[task_id] = service_start
        total_wait += max(0.0, service_start - task.release)
        cursor = finish
        loc = (task.x, task.y)
    return True, total_wait, starts


def edge_best_insertion(route: list[int], task_id: int, vehicle: Vehicle, now: float,
                        tasks: dict[int, Task]):
    best = None
    for position in range(len(route) + 1):
        candidate = route[:position] + [task_id] + route[position:]
        feasible, cost, starts = edge_route_metrics(candidate, vehicle, now, tasks)
        if feasible and (best is None or (cost, position) < (best[0], best[1])):
            best = (cost, position, candidate, starts)
    return best


def edge_best_valid_sequence(route: list[int], vehicle: Vehicle, now: float,
                             tasks: dict[int, Task]):
    if len(route) < 2:
        return route[:]
    options = exact_route_options(vehicle, route, now, tasks)
    full_mask = (1 << len(set(route))) - 1
    if full_mask not in options:
        return route[:]
    return list(options[full_mask][2])


def edge_initial_routes(task_ids: list[int], vehicles: list[Vehicle], now: float,
                        tasks: dict[int, Task]):
    """Paper-style constrained-task-first sequence insertion initialization."""
    routes = [[] for _ in vehicles]
    remaining = set(task_ids)
    while remaining:
        choices = {}
        for task_id in remaining:
            candidates = []
            task = tasks[task_id]
            for vid, vehicle in enumerate(vehicles):
                if vehicle.sensor != task.sensor:
                    continue
                insertion = edge_best_insertion(routes[vid], task_id, vehicle, now, tasks)
                if insertion is None:
                    continue
                _, old_wait, _ = edge_route_metrics(routes[vid], vehicle, now, tasks)
                candidates.append((insertion[0] - old_wait, vid, insertion))
            if candidates:
                choices[task_id] = candidates
        if not choices:
            break
        task_id = min(choices, key=lambda j: (
            len(choices[j]), tasks[j].deadline, -tasks[j].service,
            min(item[0] for item in choices[j]), j,
        ))
        _, vid, insertion = min(choices[task_id], key=lambda item: (item[0], item[1]))
        routes[vid] = insertion[2]
        remaining.remove(task_id)
    return routes, remaining


def edge_route_state(routes, vehicles, now, tasks):
    total_wait = 0.0
    completed = 0
    starts = {}
    for vid, route in enumerate(routes):
        feasible, wait, route_starts = edge_route_metrics(route, vehicles[vid], now, tasks)
        if not feasible:
            raise RuntimeError(f"infeasible Edge coalition route: vehicle={vid}, route={route}")
        starts.update(route_starts)
        total_wait += wait
        completed += len(route)
    return completed, total_wait, starts


def edge_coalition_plan(pending: set[int], vehicles: list[Vehicle], now: float,
                        tasks: dict[int, Task], seed: int):
    """Adapt the reproduced coalition-switch baseline to this event snapshot.

    The native count/wait objective and historical no-return rule are retained;
    service time, current active-job release state, completion deadlines, and
    the Manhattan road metric come from the common model.
    """
    task_ids = sorted(pending)
    routes, remaining = edge_initial_routes(task_ids, vehicles, now, tasks)
    routes = [edge_best_valid_sequence(route, vehicles[k], now, tasks)
              for k, route in enumerate(routes)]
    histories = [set() for _ in tasks]
    for vid, route in enumerate(routes):
        for task_id in route:
            histories[task_id].add(vid)

    rng = random.Random(seed + 9103)
    max_rounds = max(1, len(task_ids) * max(1, len(vehicles)))
    for _ in range(max_rounds):
        changed = False
        order = list(task_ids)
        rng.shuffle(order)
        for task_id in order:
            source = next((vid for vid, route in enumerate(routes) if task_id in route), None)
            current_count, current_wait, current_starts = edge_route_state(routes, vehicles, now, tasks)
            task = tasks[task_id]
            proposals = []
            for target, vehicle in enumerate(vehicles):
                if target == source or target in histories[task_id] or vehicle.sensor != task.sensor:
                    continue
                source_route = routes[source][:] if source is not None else None
                target_route = routes[target][:]
                if source is not None:
                    source_route.remove(task_id)
                insertion = edge_best_insertion(target_route, task_id, vehicle, now, tasks)
                if insertion is None:
                    continue
                candidate_routes = [route[:] for route in routes]
                if source is not None:
                    candidate_routes[source] = edge_best_valid_sequence(
                        source_route, vehicles[source], now, tasks
                    )
                candidate_routes[target] = edge_best_valid_sequence(
                    insertion[2], vehicles[target], now, tasks
                )
                if any(not edge_route_metrics(route, vehicles[k], now, tasks)[0]
                       for k, route in enumerate(candidate_routes)):
                    continue
                count, wait, new_starts = edge_route_state(candidate_routes, vehicles, now, tasks)
                affected = set((routes[source] if source is not None else []) + routes[target])
                harmed = any(
                    new_starts.get(other, math.inf) > current_starts.get(other, math.inf) + EPS
                    for other in affected if other != task_id and other in current_starts
                )
                if harmed:
                    continue
                if source is None:
                    system_improves = count > current_count
                    transfer_preference = task_id in new_starts
                else:
                    _, source_before, _ = edge_route_metrics(routes[source], vehicles[source], now, tasks)
                    _, source_after, _ = edge_route_metrics(candidate_routes[source], vehicles[source], now, tasks)
                    _, target_before, _ = edge_route_metrics(routes[target], vehicles[target], now, tasks)
                    _, target_after, _ = edge_route_metrics(candidate_routes[target], vehicles[target], now, tasks)
                    system_improves = count > current_count or (count == current_count and wait < current_wait - EPS)
                    transfer_preference = source_before - source_after > target_after - target_before + EPS
                if system_improves and transfer_preference:
                    proposals.append((wait, target, candidate_routes))
            if proposals:
                _, target, routes = min(proposals, key=lambda item: (item[0], item[1]))
                histories[task_id].add(target)
                changed = True
        if not changed:
            break

    assigned = {task_id for route in routes for task_id in route}
    for vid, route in enumerate(routes):
        vehicles[vid].queue.extend(route)
    pending.difference_update(assigned)
