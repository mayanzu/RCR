"""Frozen-policy fleet scaling with explicit common-load generation metadata."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import json
import time
from run_rosr_bench import save, source_hash, parse_seeds
from rosr import RosrPolicy, model, audit, verify_frozen
from rosr.comparison import DensityPolicy, JointLnsPolicy
from value_coalition import ValueCoalitionPolicy
import hashlib

ROOT=Path(__file__).resolve().parent

def worker(job):
    seed,n,method,output,seconds,compact=job
    scale=n/8
    profile=model.Profile('coupled_primary',arrival_rate=scale,edge_multiplier=1.5/scale)
    out=Path(output)/str(n)/str(seed)/method
    provenance=dict(source_sha256=source_hash(),n_vehicles=n,profile=asdict(profile),
                    seconds=seconds,compact_ties=compact)
    if method=='coalition_value':provenance['control_sha256']=hashlib.sha256((ROOT/'value_coalition.py').read_bytes()).hexdigest()
    if (out/'summary.json').exists():
        if json.loads((out/'provenance.json').read_text())!=provenance:
            raise RuntimeError('stale scale cache')
        return json.loads((out/'summary.json').read_text())
    tasks,vehicles=model.generate(seed,profile,n_vehicles=n)
    factory={'rosr':RosrPolicy,'joint_lns':JointLnsPolicy,'e2e_resource_density':DensityPolicy,'coalition_value':ValueCoalitionPolicy}.get(method)
    policy=factory(seconds=seconds,compact_ties=compact if method=='rosr' else False,seed=seed) if factory else None
    env=model.Env(tasks,vehicles,profile.edge_multiplier)
    while not env.done:
        snap=env.snapshot()
        start=time.perf_counter()
        plan=policy(snap) if policy else model.make_plan(snap,method)
        env.planning_seconds+=time.perf_counter()-start
        env.apply(plan)
        env.advance()
    summary,rows,logs=env.results()
    summary.update(seed=seed,profile=profile.name,method=method,n_vehicles=n,
                   stream_hash=model.signature(tasks,vehicles),generation=asdict(profile))
    for record in rows+logs:
        record.update(seed=seed,profile=profile.name,method=method)
    audit.check_episode(summary,rows,logs)
    audit.audit_action_logs(tasks,vehicles,profile.edge_multiplier,rows,logs)
    save(out/'provenance.json',provenance)
    save(out/'summary.json',summary)
    save(out/'tasks.json',rows)
    if policy:save(out/'decisions.json',policy.decisions)
    (out/'events.jsonl').write_text('\n'.join(json.dumps(e) for e in logs)+'\n',encoding='utf-8')
    return summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',default='20359000-20359003')
    ap.add_argument('--vehicles',default='16')
    ap.add_argument('--seconds',type=float,default=.1)
    ap.add_argument('--compact-ties',action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--label',default='continued_scale')
    args=ap.parse_args()
    verify_frozen()
    output=ROOT/'new_runs'/args.label
    save(output/'settings.json',dict(vars(args),source_sha256=source_hash()))
    jobs=[(seed,int(n),method,str(output),args.seconds,args.compact_ties)
          for n in args.vehicles.split(',') for seed in parse_seeds(args.seeds)
          for method in ('e2e_resource_density','rosr','joint_lns','edge_adapter','e2e_repack','vehicle_repack_edf','coalition_value')]
    with ProcessPoolExecutor(args.workers) as pool:
        for s in pool.map(worker,jobs):
            print(s['n_vehicles'],s['seed'],s['method'],round(s['value_rate'],5),
                  round(s['planning_ms']),flush=True)

if __name__=='__main__':main()
