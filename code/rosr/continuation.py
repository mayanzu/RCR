"""Executable continuation witnesses, never simulations of unseen arrivals."""
from .boot import EPS, from_order, model, validate_plan, action_key


class Continuation:
    def __init__(self):
        self.plan = None
        self.tasks = {}
        self.contract = None
        self.guaranteed = None

    @staticmethod
    def signature(s):
        return (s.multiplier, tuple((v.sensor, v.speed) for v in s.vehicles),
                getattr(s, 'uplink', None))

    def advance(self, s):
        if self.plan is None:
            return None, 'empty'
        if self.signature(s) != self.contract:
            return None, 'contract_changed'
        for j, task in self.tasks.items():
            if s.tasks.get(j) != task:
                return None, 'task_changed'
            if s.status.get(j) == 'expired':
                return None, 'promised_task_expired'
        routes = tuple(tuple(j for j in r if s.status[j] == 'pending')
                       for r in self.plan.routes)
        order = tuple(j for j in self.plan.edge_order
                      if s.status[j] not in ('completed', 'active_edge'))
        try:
            plan = from_order(s, routes, order, 'retained_continuation')
        except (AssertionError, KeyError):
            return None, 'witness_invalid'
        return plan, 'valid'

    @staticmethod
    def value(s, plan):
        # Locked edge work is excluded from Plan.edge_order and counted once.
        ids = {j for j, st in s.status.items() if st == 'completed'} | set(plan.edge_order)
        if s.edge_active is not None:
            j, _, end = s.edge_active
            if j in s.tasks and end <= s.tasks[j].deadline+EPS and s.status[j] != 'expired':
                ids.add(j)
        return sum(s.tasks[j].value for j in ids)

    def remember(self, s, plan):
        validate_plan(s, plan)
        self.plan = plan
        promised = set(plan.edge_order)
        if s.edge_active is not None:
            promised.add(s.edge_active[0])
        self.tasks = {j:s.tasks[j] for j in promised if j in s.tasks}
        self.contract = self.signature(s)
        self.guaranteed = self.value(s, plan)


class ActionArchive:
    """Best admissible continuation per decoded immediate action.

    Grouping does not prune route columns: identical heads can have different
    release conflicts and different feasible tails.
    """
    def __init__(self, s, rank):
        self.s, self.rank, self.plans = s, rank, {}
        self.evaluations = self.duplicates = 0

    def add(self, plan):
        self.evaluations += 1
        key = action_key(self.s, plan)
        old = self.plans.get(key)
        self.duplicates += int(old is not None)
        if old is None or self.rank(plan) < self.rank(old):
            self.plans[key] = plan

    def best(self):
        return min(self.plans.values(), key=self.rank) if self.plans else None
