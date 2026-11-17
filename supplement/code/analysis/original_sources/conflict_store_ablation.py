"""Audited, paired re-run isolating cross-event certificate reuse.

This experiment never edits the frozen policy sources. It compares, on the
same generated grid streams and in the same run environment:

* plain_single_event: no floor lowering; discard persistent conflicts at each
  new physical event, while retaining any certificates within that event;
* plain_cross_event: the same plain cut, with the normal cross-event store;
* strengthened_cross_event: the normal strengthened cut and store.

Unlike capacity=0, explicitly clearing the existing store avoids Python's
``entries[-0:]`` behaviour, which would retain every entry. Each triplet is
run sequentially inside one worker; worker processes may still contend for
hardware. Outputs include source snapshots, fingerprints and audited traces.
Run from the project root, for example:

    python -B paper/analysis/conflict_store_ablation.py --tag reuse_pilot --count 2 --workers 1
    python -B paper/analysis/conflict_store_ablation.py --tag reuse_confirm --count 30 --workers 3
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / 'code'
sys.path.insert(0, str(CODE))

import numpy as np
import scipy
from rosr import RosrPolicy, model, verify_frozen
from rosr.boot import audit
from rosr.edge import EdgeOracle
from run_rosr_bench import source_hash

CONDITIONS = ('coupled_primary', 'edge_dominant', 'tighter_deadlines')
VARIANTS = ('plain_single_event', 'plain_cross_event', 'strengthened_cross_event')
ORIGINAL_STRENGTHEN = EdgeOracle.strengthen


class EventLocalPolicy(RosrPolicy):
    """Clear stored proofs at the boundary, not during one decision event."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.boundary_clears = 0
        self.entries_discarded = 0

    def __call__(self, state):
        if state.event_index != self.last_event or state.now != self.last_now:
            self.boundary_clears += 1
            self.entries_discarded += len(self.conflicts.entries)
            self.conflicts.entries.clear()
        return super().__call__(state)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(out: Path, seeds: list[int], workers: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(CODE.rglob('*.py')) + sorted(CODE.glob('*.json')) + [CODE/'requirements.txt', Path(__file__).resolve()]
    files = [p for p in files if '__pycache__' not in p.parts]
    manifest = dict(protocol='cross_event_conflict_ablation_v1', seeds=seeds,
                    conditions=list(CONDITIONS), variants=list(VARIANTS), workers=workers,
                    iteration_order='cyclic within each paired condition/seed triplet',
                    thread_limits={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
                    physical_core_affinity='not set', python=sys.version, platform=platform.platform(),
                    numpy=np.__version__, scipy=scipy.__version__, policy_sha256=source_hash(),
                    files={p.relative_to(ROOT).as_posix():file_hash(p) for p in files})
    metadata = out/'source_manifest.json'
    if metadata.exists():
        if json.loads(metadata.read_text(encoding='utf-8')) != manifest:
            raise RuntimeError('Existing run has different sources or settings; use a new tag.')
        return manifest
    with zipfile.ZipFile(out/'source_snapshot.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
    metadata.write_text(json.dumps(manifest,indent=2)+'\n', encoding='utf-8')
    return manifest


def save_gzip_json(path: Path, data) -> None:
    with gzip.open(path,'wt',encoding='utf-8',newline='\n') as out:
        json.dump(data,out,ensure_ascii=False,default=str)


def run_triplet(job: tuple) -> list[dict]:
    condition, seed, outdir = job
    profile = next(p for p in model.PROFILES if p.name == condition)
    order = list(VARIANTS)
    shift = (seed + CONDITIONS.index(condition)) % len(order)
    order = order[shift:] + order[:shift]
    records=[]
    for variant in order:
        folder=Path(outdir)/condition/str(seed)/variant
        if (folder/'summary.json').exists():
            records.append(json.loads((folder/'summary.json').read_text(encoding='utf-8')))
            continue
        counter={'calls':0}
        def no_strengthening(self, floors, stop=None, trials=6):
            counter['calls'] += 1
            return dict(floors)
        EdgeOracle.strengthen = ORIGINAL_STRENGTHEN if variant == 'strengthened_cross_event' else no_strengthening
        policy = (EventLocalPolicy if variant == 'plain_single_event' else RosrPolicy)(seed=seed)
        try:
            started=time.perf_counter()
            summary, task_rows, events=model.run_episode(seed, profile, variant, policy=policy)
            wall=time.perf_counter()-started
            tasks, vehicles=model.generate(seed, profile)
            audit.check_episode(summary, task_rows, events)
            audit.audit_action_logs(tasks, vehicles, profile.edge_multiplier, task_rows, events)
        finally:
            EdgeOracle.strengthen=ORIGINAL_STRENGTHEN
        result=dict(variant=variant, profile=condition, seed=seed, stream_hash=summary['stream_hash'],
                    value_rate=summary['value_rate'], completed_value=summary['completed_value'],
                    total_value=summary['total_value'], planning_ms=summary['planning_ms'],
                    count_rate=summary['count_rate'], wall_seconds=wall, audited=True,
                    policy_stats=dict(vars(policy.stats)),
                    strengthen_calls_suppressed=counter['calls'],
                    boundary_clears=getattr(policy,'boundary_clears',None),
                    entries_discarded=getattr(policy,'entries_discarded',None),
                    event_count=len(events))
        folder.mkdir(parents=True,exist_ok=True)
        save_gzip_json(folder/'tasks.json.gz',task_rows)
        save_gzip_json(folder/'events.json.gz',events)
        save_gzip_json(folder/'decisions.json.gz',policy.decisions)
        (folder/'summary.json').write_text(json.dumps(result,indent=2,default=str)+'\n',encoding='utf-8')
        records.append(result)
    if len({r['stream_hash'] for r in records}) != 1:
        raise RuntimeError(f'Stream mismatch: {condition}, {seed}')
    return records


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',required=True)
    parser.add_argument('--count',type=int,default=30)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if not args.tag.replace('_','').isalnum() or not 1 <= args.count <= 30 or args.workers < 1:
        parser.error('tag must be alphanumeric/underscore; 1 <= count <= 30; workers >= 1')
    seeds=list(range(20423000,20423000+args.count))
    out=ROOT/'opt'/'runs'/args.tag
    verify_frozen()
    freeze(out,seeds,args.workers)
    jobs=[(c,s,str(out)) for c in CONDITIONS for s in seeds]
    print(f'Starting {len(jobs)} paired triplets; {args.workers} worker(s).',flush=True)
    done=0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_triplet,job) for job in jobs]
        for future in as_completed(futures):
            future.result()
            done += 1
            print(f'Completed {done}/{len(jobs)} triplets',flush=True)
    print('All episodes completed and passed physical/action audits.',flush=True)


if __name__=='__main__':
    main()
