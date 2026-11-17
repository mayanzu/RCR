"""Paired benchmark: ROSR vs the frozen baselines over seeds x profiles.

    python -B scripts/run_rosr_bench.py --label bench_v1 \
        --seeds 20351000-20351009 --profiles coupled_primary

Episodes are independent, so they are run in a process pool (one episode per
core) and written to per-episode files. Every episode is audited with the
frozen checker, and paired deltas with descriptive intervals are computed at
the end. Existing episodes are reused, so an interrupted run resumes. Nothing
in the current runtime directory is modified.
"""
from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[1]/'code'))
from bootstrap import BASELINES_ROOT,output_directory,source_hash

import argparse
import csv
import json
import hashlib
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# One thread per worker: the pool supplies the parallelism, and oversubscribed
# BLAS/HiGHS threads would only add contention. Bytecode caches are suppressed
# to keep the current source tree free of caches.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.dont_write_bytecode = True


import numpy as np
from scipy.stats import t
import platform
from dataclasses import asdict
from rosr.comparison import DensityPolicy, JointLnsPolicy
from rosr import RosrPolicy, model, verify_frozen  # noqa: E402
from rosr.boot import audit  # noqa: E402
from baselines.value_coalition import ValueCoalitionPolicy
from baselines.lit_dpso import DpsoPolicy
from baselines.lit_lyapunov import LyapunovPolicy

def interval(values):
    a = np.asarray(values, dtype=float)
    n = len(a)
    mean = float(a.mean()) if n else None
    radius = float(t.ppf(.975, n-1)*a.std(ddof=1)/math.sqrt(n)) if n > 1 else None
    return dict(n=n, mean=mean, ci95_lo=mean-radius if radius is not None else None,
                ci95_hi=mean+radius if radius is not None else None)

BASELINE = 'e2e_resource_density'
METHODS = (BASELINE, 'rosr', 'joint_lns', 'edge_adapter', 'e2e_repack', 'vehicle_repack_edf','coalition_value',
           'lit_lyapunov', 'lit_dpso')
METRICS = ('value_rate', 'count_rate', 'downstream_lost_value_rate',
           'uncaptured_lost_value_rate', 'wasted_capture_time',
           'edge_wait_time', 'capture_time', 'planning_ms')


def parse_seeds(spec: str):
    out = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            lo, hi = part.split('-', 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def run_one(seed, profile_name, method, outdir, pol_kwargs):
    """Run (or load) one episode. Runs inside a worker process."""
    profile = next(p for p in model.PROFILES if p.name == profile_name)
    ep = Path(outdir) / str(seed) / profile_name / method
    fingerprint = dict(source_sha256=source_hash(), policy_kwargs=pol_kwargs)
    if method=='coalition_value':fingerprint['control_sha256']=hashlib.sha256((BASELINES_ROOT/'value_coalition.py').read_bytes()).hexdigest()
    if method=='lit_dpso':fingerprint['control_sha256']=hashlib.sha256((BASELINES_ROOT/'lit_dpso.py').read_bytes()).hexdigest()
    if method=='lit_lyapunov':fingerprint['control_sha256']=hashlib.sha256((BASELINES_ROOT/'lit_lyapunov.py').read_bytes()).hexdigest()
    if (ep / 'summary.json').exists():
        if not (ep / 'provenance.json').exists() or json.loads(
                (ep / 'provenance.json').read_text(encoding='utf-8')) != fingerprint:
            raise RuntimeError(f'Cached code/settings mismatch: {ep}; use a new output folder')
        return json.loads((ep / 'summary.json').read_text(encoding='utf-8')), False
    tasks, vehicles = model.generate(seed, profile)
    policy = None
    if method == 'rosr':
        policy = RosrPolicy(seed=seed, **pol_kwargs)
    elif method == BASELINE:
        policy = DensityPolicy()
    elif method == 'coalition_value':
        policy=ValueCoalitionPolicy(**pol_kwargs)
    elif method == 'joint_lns':
        lns_kwargs = dict(pol_kwargs)
        lns_kwargs['compact_ties'] = False
        policy = JointLnsPolicy(**lns_kwargs)
    elif method == 'lit_lyapunov':
        policy = LyapunovPolicy(seed=seed, **{k: v for k, v in pol_kwargs.items() if k in ('seconds',)})
    elif method == 'lit_dpso':
        policy = DpsoPolicy(seed=seed, **{k: v for k, v in pol_kwargs.items() if k in ('seconds',)})
    summary, rows, logs = model.run_episode(seed, profile, method, policy=policy)
    audit.check_episode(summary, rows, logs)
    audit.audit_action_logs(tasks, vehicles, profile.edge_multiplier, rows, logs)
    if policy is not None:
        stats = policy.stats.__dict__ if hasattr(policy.stats, '__dict__') else dict(policy.stats)
        summary = dict(summary, policy_stats=stats)
        save(ep / 'decisions.json', policy.decisions)
    save(ep / 'provenance.json', fingerprint)
    save(ep / 'summary.json', summary)
    save(ep / 'tasks.json', rows)
    (ep / 'events.jsonl').parent.mkdir(parents=True, exist_ok=True)
    (ep / 'events.jsonl').write_text('\n'.join(json.dumps(e) for e in logs) + '\n', encoding='utf-8')
    return summary, True


def _worker(job):
    seed, profile_name, methods, outdir, pol_kwargs = job
    shift=(seed+next(i for i,p in enumerate(model.PROFILES) if p.name==profile_name))%len(methods)
    order=methods[shift:]+methods[:shift]
    rows=[]
    for method in order:
        started = time.perf_counter()
        summary, ran = run_one(seed, profile_name, method, outdir, pol_kwargs)
        rows.append((seed, profile_name, method, summary, ran, time.perf_counter()-started))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='rosr_bench')
    ap.add_argument('--seeds', default='20351000-20351009')
    ap.add_argument('--profiles', default=','.join(p.name for p in model.PROFILES))
    ap.add_argument('--methods', default=','.join(METHODS))
    ap.add_argument('--output', default=None, help='default ../VCS-E2E_runs/<label>')
    ap.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument('--beta', type=float, default=None)
    ap.add_argument('--max-depth', type=int, default=None)
    ap.add_argument('--seconds', type=float, default=None)
    ap.add_argument('--compact-ties', action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument('--conflict-cuts', action=argparse.BooleanOptionalAction, default=None)
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    names = [p.strip() for p in args.profiles.split(',') if p.strip()]
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    if not methods or set(methods)-set(METHODS):
        ap.error('methods must use canonical names: '+','.join(METHODS))
    if not names or set(names)-{p.name for p in model.PROFILES}:
        ap.error('unknown or empty profile selection')
    profiles = [p for p in model.PROFILES if p.name in names]
    outdir = output_directory(args.output,label=args.label)
    pol_kwargs = {}
    for key, val in (('beta', args.beta),
                     ('max_depth', args.max_depth), ('seconds', args.seconds),
                     ('compact_ties', args.compact_ties), ('conflict_cuts', args.conflict_cuts)):
        if val is not None:
            pol_kwargs[key] = val

    hashes = verify_frozen()
    settings = dict(seeds=seeds, profiles=[p.name for p in profiles], methods=methods,
                    policy_kwargs=pol_kwargs, frozen_hashes=len(hashes), source_sha256=source_hash(),
                    workers=args.workers, hardware=dict(platform=platform.platform(),processor=platform.processor(),
                    cpu_count=os.cpu_count(),python=platform.python_version(),
                    thread_env={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}), started=time.strftime('%Y-%m-%d %H:%M:%S'))
    save(outdir / 'settings.json', settings)

    settings['method_order']='rotated within each seed/profile worker; one fresh environment per method'
    save(outdir/'settings.json',settings)
    jobs = [(seed, p.name, tuple(methods), str(outdir), pol_kwargs)
            for p in profiles for seed in seeds]
    results = {}
    wall_started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, job): job for job in jobs}
        for fut in as_completed(futures):
            for seed, prof, method, summary, ran, took in fut.result():
                results[seed, prof, method] = summary
                print(f'{seed} {prof} {method} J={summary["value_rate"]:.4f} '
                      f'down={summary["downstream_lost_value_rate"]:.4f} '
                      f'plan_ms={summary["planning_ms"]:.0f} '
                      f'{"ran" if ran else "cached"} {took:.1f}s', flush=True)

    # Paired comparisons against the frozen baseline on identical streams.
    comparisons = []
    for profile in profiles:
        for method in methods:
            if method == BASELINE:
                continue
            seeds_ok = [s for s in seeds
                        if (s, profile.name, method) in results and (s, profile.name, BASELINE) in results
                        and results[s, profile.name, method]['stream_hash']
                        == results[s, profile.name, BASELINE]['stream_hash']]
            row = {'profile': profile.name, 'method': method, 'n': len(seeds_ok)}
            for metric in METRICS:
                delta = [100.0 * (results[s, profile.name, method][metric]
                                  - results[s, profile.name, BASELINE][metric])
                         if metric.endswith('_rate') else
                         (results[s, profile.name, method][metric]
                          - results[s, profile.name, BASELINE][metric])
                         for s in seeds_ok]
                row[metric] = interval(delta) if delta else {}
            vd = [results[s, profile.name, method]['value_rate']
                  - results[s, profile.name, BASELINE]['value_rate'] for s in seeds_ok]
            row['wins'] = sum(1 for x in vd if x > 1e-12)
            row['ties'] = sum(1 for x in vd if abs(x) <= 1e-12)
            row['losses'] = sum(1 for x in vd if x < -1e-12)
            row['value_rate_pp'] = interval([100.0 * x for x in vd]) if vd else {}
            comparisons.append(row)
    save(outdir / 'comparisons.json', comparisons)

    csv_path = outdir / 'episodes.csv'
    fields = ['profile', 'seed', 'method', *METRICS, 'stream_hash']
    with csv_path.open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for (seed, prof, method), s in sorted(results.items()):
            w.writerow({'profile': prof, 'seed': seed, 'method': method,
                        **{k: s[k] for k in METRICS}, 'stream_hash': s['stream_hash']})

    print(json.dumps([{k: v for k, v in c.items() if k in
                       ('profile', 'method', 'n', 'wins', 'ties', 'losses', 'value_rate_pp')}
                      for c in comparisons], indent=2))
    print(f'wall {time.perf_counter() - wall_started:.1f}s -> wrote {outdir}')


if __name__ == '__main__':
    main()
