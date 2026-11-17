"""Equivalent O(n log n) witnesses when every report is required."""
import heapq
from .boot import EPS, model


class RequiredScheduler:
    def __init__(self, s):
        self.base = model.edge_cursor(s)
        self.work = {j: t.work*s.multiplier for j, t in s.tasks.items()}
        self.deadline = {j: t.deadline for j, t in s.tasks.items()}
        self.density = {j: (-t.value/self.work[j], t.deadline, j) for j, t in s.tasks.items()}

    def schedule(self, releases, rule, margin=0.):
        """Same priorities, releases and EPS as model.edge_schedule.

        A missed job is mandatory, so failure can return immediately instead
        of finishing an order that will necessarily fail its required-set test.
        """
        cursor = self.base
        times, order = {}, []
        if rule == 'reserved_edf' or all(r <= cursor+EPS for r in releases.values()):
            key = ((lambda j: self.density[j]) if rule == 'density'
                   else (lambda j: (self.deadline[j], releases[j], j)))
            sequence = sorted(releases, key=key)
            for j in sequence:
                start = max(cursor, releases[j])
                end = start+self.work[j]
                if end > self.deadline[j]-margin+EPS:
                    return None
                times[j] = (start, end)
                order.append(j)
                cursor = end
            return tuple(order), times
        future = sorted(releases, key=lambda j: (releases[j], j))
        queue, pos = [], 0
        while pos < len(future) or queue:
            if not queue:
                cursor = max(cursor, releases[future[pos]])
            while pos < len(future) and releases[future[pos]] <= cursor+EPS:
                j = future[pos]
                priority = self.density[j] if rule == 'density' else (self.deadline[j], releases[j], j)
                heapq.heappush(queue, (priority, j))
                pos += 1
            _, j = heapq.heappop(queue)
            start = max(cursor, releases[j])
            end = start+self.work[j]
            if end > self.deadline[j]-margin+EPS:
                return None
            times[j] = (start, end)
            order.append(j)
            cursor = end
        return tuple(order), times
