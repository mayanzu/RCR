"""Shared release-aware nonpreemptive oracle for packing and search controls."""
from __future__ import annotations
from collections import Counter
from .boot import EPS, model
from .scheduling import RequiredScheduler

EXACT_MAX = 14


class ConflictStore:
    """Bounded cross-event memory of infeasibility, never of feasible plans."""
    def __init__(self, capacity=64):
        self.capacity = capacity
        self.entries = []
        self.hits = 0

    def active(self, s, available=None):
        available = set(s.tasks) if available is None else set(available)
        base = model.edge_cursor(s)
        out = []
        for floors, attributes, proved_base in self.entries:
            if (set(floors) <= available and base >= proved_base
                    and all((s.tasks[j].work*s.multiplier, s.tasks[j].deadline) == attributes[j]
                            for j in floors)):
                out.append(floors)
        return out

    def match(self, s, releases):
        for floors in self.active(s, releases):
            if all(max(model.edge_cursor(s), releases[j]) >= r for j, r in floors.items()):
                self.hits += 1
                return floors
        return None

    def add(self, s, floors):
        if not floors:
            return
        attrs = {j: (s.tasks[j].work*s.multiplier, s.tasks[j].deadline) for j in floors}
        base = model.edge_cursor(s)
        # Componentwise smaller floors and an earlier proof base dominate.
        for old, old_attrs, old_base in self.entries:
            if (old.keys() == floors.keys() and attrs == old_attrs and old_base <= base
                    and all(old[j] <= floors[j] for j in floors)):
                return
        self.entries = [(old, a, t) for old, a, t in self.entries
                        if not (old.keys() == floors.keys() and attrs == a and base <= t
                                and all(floors[j] <= old[j] for j in floors))]
        self.entries.append((dict(floors), attrs, base))
        self.entries = self.entries[-self.capacity:]


class EdgeOracle:
    def __init__(self, s, exact_max=EXACT_MAX, store=None, margin=0.):
        self.s, self.exact_max = s, exact_max
        self.store = store if margin == 0. else None
        self.margin = margin
        self.scheduler = RequiredScheduler(s)
        self.cache = {}
        self.exact_cache = {}
        # Certified-infeasible subset of each infeasible set. Above the exact
        # cap this is the local busy window, which keeps conflict extraction
        # as small as the proof that produced it.
        self.proof_subset = {}
        self.stats = Counter()

    def _improve(self, order, releases):
        """Light adjacent-swap pass: same jobs, earlier total completion.

        Every trial keeps all deadlines met, so the result stays a feasibility
        witness. Among the three linear witnesses this lets the decoder keep
        the one that frees the shared processor soonest.
        """
        s = self.s
        work = {j: s.tasks[j].work * s.multiplier for j in order}

        def times(seq):
            cursor, total, out = model.edge_cursor(s), 0.0, {}
            for j in seq:
                end = max(cursor, releases[j]) + work[j]
                if end > s.tasks[j].deadline - self.margin + EPS:
                    return None
                out[j] = end
                total += end
                cursor = end
            return total
        best = list(order)
        current = times(best)
        if current is None:
            return tuple(best)
        improved = True
        while improved:
            improved = False
            for i in range(len(best) - 1):
                trial = best[:i] + [best[i + 1], best[i]] + best[i + 2:]
                score = times(trial)
                if score is not None and score < current - 1e-12:
                    best, current, improved = trial, score, True
                    break
        return tuple(best)

    def _busy_window(self, releases, cap):
        """Local competition subset around the first deadline overrun.

        Nonpreemptive infeasibility is local in time: the jobs that push the
        first failing deadline past its limit all lie in the busy block that
        contains that failure. Any subset is sound for a proof, since subset
        infeasibility implies infeasibility of the whole set; the window is a
        search heuristic for finding such a subset, not a claim of minimality.
        """
        s = self.s
        cursor = model.edge_cursor(s)
        pending = dict(releases)
        block = []
        while pending:
            available = [j for j in pending if pending[j] <= cursor + EPS]
            if not available:
                cursor = min(pending.values())
                block = []
                continue
            j = min(available, key=lambda k: (s.tasks[k].deadline, k))
            end = cursor + s.tasks[j].work * s.multiplier
            if end > s.tasks[j].deadline - self.margin + EPS:
                candidates = set(block) | {j}
                candidates |= {k for k in pending
                               if s.tasks[k].deadline <= s.tasks[j].deadline + EPS}
                if len(candidates) > cap:
                    # Keep the jobs nearest the failing deadline, breaking ties
                    # toward the heaviest processing demand: those are the
                    # tasks that actually consume the contested interval.
                    nearest = sorted(candidates,
                                     key=lambda k: (s.tasks[k].deadline,
                                                    s.tasks[k].work * s.multiplier,
                                                    k))[-(cap - 1):]
                    candidates = set(nearest) | {j}
                return {k: releases[k] for k in candidates}
            block.append(j)
            cursor = end
            del pending[j]
        return None

    def fixed(self, releases):
        """Serve every supplied job, returning (order, proved_infeasible).

        A greedy success is a feasibility witness, so exponential search is
        unnecessary. Exhaustive subset failure on the whole set, or on a local
        busy-window subset, is an infeasibility proof; only a window that is
        itself feasible leaves a large instance unknown.
        """
        self.stats['calls'] += 1
        key = tuple(sorted(releases.items()))
        if key in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[key]
        conflict = self.store.match(self.s, releases) if self.store is not None else None
        if conflict is not None:
            self.stats['certificate_hits'] += 1
            self.proof_subset[key] = {j: releases[j] for j in conflict}
            self.cache[key] = (None, True)
            return None, True
        required = set(releases)
        candidates = []
        for rule in ('edf', 'reserved_edf', 'density'):
            result = self.scheduler.schedule(releases, rule, margin=self.margin)
            if result is not None:
                order, times = result
                if any(end > self.s.tasks[j].deadline-self.margin+EPS
                       for j, (_, end) in times.items()):
                    continue
                candidates.append((sum(end for _, end in times.values()),
                                   max((end for _, end in times.values()), default=0), order))
        if candidates:
            self.stats['witness'] += 1
            result = self._improve(min(candidates)[2], releases), False
        elif len(releases) <= self.exact_max:
            self.stats['exact_calls'] += 1
            order = self._exact_all(releases)
            self.stats['exact_infeasible' if order is None else 'exact_witness'] += 1
            result = (None if order is None else self._improve(order, releases)), order is None
            if order is None:
                self.proof_subset[key] = dict(releases)
        else:
            window = self._busy_window(releases, self.exact_max)
            order = self._exact_all(window) if window else None
            if window is not None and order is None:
                self.stats['local_proof'] += 1
                self.proof_subset[key] = window
                result = None, True
            else:
                self.stats['unknown'] += 1
                result = None, False
        self.cache[key] = result
        return result

    def _exact_all(self, releases):
        """Exact subset DP; identical semantics, tighter inner loop.

        Results are memoized per release vector so deletion minimization and
        floor strengthening share work instead of re-solving the same subset.
        """
        key = tuple(sorted(releases.items()))
        hit = self.exact_cache.get(key)
        if hit is not None:
            self.stats['exact_cache_hits'] += 1
            return hit
        s = self.s
        ids = sorted(releases)
        n, size = len(ids), 1 << len(ids)
        times = [float('inf')] * size
        paths = [None] * size
        times[0], paths[0] = model.edge_cursor(s), ()
        work = [s.tasks[j].work * s.multiplier for j in ids]
        deadlines = [s.tasks[j].deadline-self.margin for j in ids]
        ready = [releases[j] for j in ids]
        bits = [1 << k for k in range(n)]
        full = size - 1
        EPS_ = EPS
        for mask in range(size):
            path = paths[mask]
            if path is None:
                continue
            cursor = times[mask]
            for k in range(n):
                bit = bits[k]
                if mask & bit:
                    continue
                rk = ready[k]
                end = (cursor if cursor >= rk else rk) + work[k]
                if end > deadlines[k] + EPS_:
                    continue
                new = mask | bit
                newpath = path + (ids[k],)
                old = times[new]
                if end < old or (end == old and (paths[new] is None or newpath < paths[new])):
                    times[new] = end
                    paths[new] = newpath
        result = paths[full]
        self.exact_cache[key] = result
        return result

    def minimal_conflict(self, releases, stop=None):
        """Deletion-minimal infeasible fixed-release subset, never an unknown.

        Deletion starts from the certified subset of the proof: when the whole
        set exceeds the exact cap, the local busy window already carries the
        certificate, and searching outside it cannot shrink a conflict that
        lives inside it.
        """
        key = tuple(sorted(releases.items()))
        order, proof = self.fixed(releases)
        if not proof:
            raise ValueError('a certified infeasible set is required')
        conflict = dict(self.proof_subset.get(key, releases))
        for j in sorted(conflict):
            if stop is not None and stop():
                break
            reduced = {k: r for k, r in conflict.items() if k != j}
            _, proof = self.fixed(reduced)
            if proof:
                conflict = reduced
        return conflict

    def strengthen(self, floors, stop=None, trials=6):
        """Lower a few release floors, retaining only exact infeasibility proofs."""
        strengthened = dict(floors)
        base = model.edge_cursor(self.s)
        used = 0
        for j in sorted(floors, key=lambda j: (-floors[j], j)):
            if used >= trials or (stop is not None and stop()):
                break
            if strengthened[j] <= base:
                continue
            trial = dict(strengthened)
            trial[j] = base
            used += 1
            if self._exact_all(trial) is None:
                strengthened = trial
                continue
            # One midpoint trial controls certification overhead.
            if used >= trials or (stop is not None and stop()):
                break
            trial[j] = (base+strengthened[j])/2
            used += 1
            if self._exact_all(trial) is None:
                strengthened = trial
        self.stats['floor_trials'] += used
        return strengthened

    def select(self, releases, required):
        """Shared fixed-route value selection used by the joint search control."""
        s = self.s
        required = set(required)
        plans = []
        for rule in ('edf', 'reserved_edf', 'density'):
            result = model.edge_schedule(s, releases, rule, required)
            if result is not None:
                order, times = result
                plans.append((-sum(s.tasks[j].value for j in order), -len(order),
                              sum(end - s.tasks[j].release for j, (_, end) in times.items()), order))
        best = min(plans) if plans else None
        total = sum(s.tasks[j].value for j in releases)
        # Give the joint control the same fixed-set feasibility oracle and cap
        # as route packing, before solving optional report selection.
        if best is None or -best[0] < total - EPS:
            full_order, _ = self.fixed(releases)
            if full_order is not None:
                return full_order
        if len(releases) <= 12 and (best is None or -best[0] < total - EPS):
            self.stats['selection_dp'] += 1
            result = model.exact_edge(s, releases, required=required, max_jobs=12)
            if result is not None:
                order, times = result
                candidate = (-sum(s.tasks[j].value for j in order), -len(order),
                             sum(end-s.tasks[j].release for j, (_, end) in times.items()), order)
                if best is None or candidate < best:
                    best = candidate
        if best is None and required:
            order, _ = self.fixed({j:releases[j] for j in required})
            if order is not None:
                return self._improve(order, releases)
        return None if best is None else self._improve(best[-1], releases)
