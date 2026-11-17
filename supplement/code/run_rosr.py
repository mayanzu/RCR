"""Run one episode under the ROSR policy and report value, timing and decisions.

    python -B code/run_rosr.py --seed 20351000 --profile coupled_primary \
        --output code/new_runs/rosr_smoke_01

Compares against the frozen e2e_resource_density baseline on the same stream.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rosr import RosrPolicy, model, verify_frozen  # noqa: E402
from rosr.comparison import DensityPolicy
from rosr.boot import audit  # noqa: E402


def run(seed, profile_name, policy_factory, label):
    profile = next(p for p in model.PROFILES if p.name == profile_name)
    started = time.perf_counter()
    summary, rows, logs = model.run_episode(seed, profile, label, policy=policy_factory())
    wall = time.perf_counter() - started
    return summary, rows, logs, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=20351000)
    ap.add_argument('--profile', default='coupled_primary')
    ap.add_argument('--output', default=None)
    ap.add_argument('--beta', type=float, default=None)
    ap.add_argument('--max-depth', type=int, default=None)
    ap.add_argument('--seconds', type=float, default=None)
    args = ap.parse_args()

    hashes = verify_frozen()
    print(json.dumps({'frozen_hashes_verified': len(hashes)}))

    kw = {}
    for key, val in (('beta', args.beta), 
                     ('max_depth', args.max_depth), ('seconds', args.seconds)):
        if val is not None:
            kw[key] = val

    policies = {}

    def make_rosr():
        p = RosrPolicy(seed=args.seed, **kw)
        policies['rosr'] = p
        return p

    base, base_rows, base_logs, base_wall = run(
        args.seed, args.profile, lambda: DensityPolicy(), 'e2e_resource_density')
    ros, ros_rows, ros_logs, ros_wall = run(args.seed, args.profile, make_rosr, 'rosr')

    # Raw audit of both episodes against the frozen checker.
    for s, rows, logs in ((base, base_rows, base_logs), (ros, ros_rows, ros_logs)):
        profile = next(p for p in model.PROFILES if p.name == s['profile'])
        tasks, vehicles = model.generate(s['seed'], profile)
        audit.check_episode(s, rows, logs)
        audit.audit_action_logs(tasks, vehicles, profile.edge_multiplier, rows, logs)

    pol = policies['rosr']
    report = {
        'seed': args.seed, 'profile': args.profile,
        'stream_hash': base['stream_hash'],
        'stream_hashes_equal': base['stream_hash'] == ros['stream_hash'],
        'baseline_value_rate': base['value_rate'],
        'rosr_value_rate': ros['value_rate'],
        'delta_pp': 100.0 * (ros['value_rate'] - base['value_rate']),
        'baseline_downstream_lost': base['downstream_lost_value_rate'],
        'rosr_downstream_lost': ros['downstream_lost_value_rate'],
        'baseline_wasted_capture_min': base['wasted_capture_time'],
        'rosr_wasted_capture_min': ros['wasted_capture_time'],
        'baseline_wall_s': base_wall, 'rosr_wall_s': ros_wall,
        'baseline_planning_ms': base['planning_ms'],
        'rosr_planning_ms': ros['planning_ms'],
        'audited': True,
        'rosr_stats': dict(pol.stats.__dict__),
    }
    print(json.dumps(report, indent=2))

    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        (out / 'tasks.json').write_text(json.dumps(ros_rows, indent=2), encoding='utf-8')
        (out / 'events.jsonl').write_text(
            '\n'.join(json.dumps(e) for e in ros_logs), encoding='utf-8')
        (out / 'decisions.json').write_text(
            json.dumps(pol.decisions, indent=2, default=str), encoding='utf-8')
        print(f'wrote {out}')


if __name__ == '__main__':
    main()
