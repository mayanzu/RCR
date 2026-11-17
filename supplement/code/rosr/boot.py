"""Current simulator, routing dependencies and shared plan validation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / 'runtime'
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from common import action_key, audit, model, verify_frozen  # noqa: E402

EPS = model.EPS
from .transmission import report_releases  # noqa: E402


def validate_plan(s, p, pred=None, releases=None):
    if pred is None:
        pred = model.route_predictions(s,p.routes)
    assert pred is not None
    assert len(p.edge_order)==len(set(p.edge_order))
    assert set(pred[0]).issubset(p.edge_order), 'an acquisition must have a feasible report'
    if releases is None:
        releases = report_releases(s,pred[0],p.routes)
    cursor = model.edge_cursor(s)
    for j in p.edge_order:
        assert j in releases
        start=max(cursor,releases[j]); end=start+s.tasks[j].work*s.multiplier
        assert end<=s.tasks[j].deadline+EPS
        assert abs(p.edge_times[j][0]-start)<EPS and abs(p.edge_times[j][1]-end)<EPS
        cursor=end
    assert abs(p.value-sum(s.tasks[j].value for j in p.edge_order))<EPS
    return True


def from_order(s,routes,order,rule):
    pred=model.route_predictions(s,routes)
    assert pred is not None
    captures,travel=pred
    releases=report_releases(s,captures,routes)
    cursor,times=model.edge_cursor(s),{}
    for j in order:
        start=max(cursor,releases[j]); end=start+s.tasks[j].work*s.multiplier
        assert end<=s.tasks[j].deadline+EPS
        times[j]=(start,end);cursor=end
    p=model.Plan(tuple(tuple(r) for r in routes),tuple(order),times,captures,
        sum(s.tasks[j].value for j in order),len(order),
        sum(end-s.tasks[j].release for j,(_,end) in times.items()),travel,rule)
    # Predictions above are exactly what validate_plan recomputes; pass them
    # through so every decode does one prediction, not two.
    validate_plan(s,p,pred=pred,releases=releases)
    return p

__all__ = ['ROOT', 'SIM', 'EPS', 'model', 'audit', 'verify_frozen', 'action_key',
           'validate_plan', 'from_order', 'report_releases']
