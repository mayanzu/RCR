"""Replay 270 saved grid episodes against physical/action audits.

Run from the project root: python -B paper/analysis/verify_conflict_store.py
This verifies saved traces against the current fingerprinted model; it does
not infer provenance of the manuscript's older grid summary files.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'opt/runs/eswa_certificate_confirmation_20261112'
sys.path.insert(0,str(ROOT/'code'))
from rosr import model
from rosr.boot import audit
from run_rosr_bench import source_hash


def main():
    manifest=json.loads((RUN/'source_manifest.json').read_text(encoding='utf8'))
    assert source_hash()==manifest['policy_sha256']
    for relative,expected in manifest['files'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected,relative
    verified=0
    for folder in sorted(p.parent for p in RUN.glob('*/*/*/summary.json')):
        saved=json.loads((folder/'summary.json').read_text(encoding='utf8'))
        with gzip.open(folder/'tasks.json.gz','rt',encoding='utf8') as f: rows=json.load(f)
        with gzip.open(folder/'events.json.gz','rt',encoding='utf8') as f: events=json.load(f)
        profile=next(p for p in model.PROFILES if p.name==saved['profile'])
        tasks,vehicles=model.generate(saved['seed'],profile)
        total=sum(r['value'] for r in rows)
        completed=[r for r in rows if r['status']=='completed']
        captured=[r for r in rows if r['capture_finish'] is not None]
        expired_captured=[r for r in captured if r['status']!='completed']
        ratio=lambda group:sum(r['value'] for r in group)/total if total else 0.
        horizon=model.H
        before=lambda a,b: sum(max(0.,min(r[b],horizon)-max(r[a],0.))
                                for r in rows if r[a] is not None and r[b] is not None)
        summary=dict(total_value=total,value_rate=ratio(completed),capture_value_rate=ratio(captured),
                     downstream_lost_value_rate=ratio(expired_captured),
                     uncaptured_lost_value_rate=1.-ratio(captured) if total else 0.,
                     vehicle_utilization_0_h=before('capture_start','capture_finish')/(len(vehicles)*horizon),
                     edge_utilization_0_h=before('edge_start','edge_finish')/horizon)
        assert abs(saved['total_value']-total)<1e-9
        assert abs(saved['completed_value']-sum(r['value'] for r in completed))<1e-9
        assert abs(saved['value_rate']-summary['value_rate'])<1e-12
        assert saved['stream_hash']==model.signature(tasks,vehicles)
        audit.check_episode(summary,rows,events)
        audit.audit_action_logs(tasks,vehicles,profile.edge_multiplier,rows,events)
        if saved['variant']=='plain_single_event':
            assert saved['boundary_clears']>0 and saved['policy_stats']['reused_conflict_cuts']==0
        verified+=1
    assert verified==270,verified
    report={'episodes_replayed':verified,'matching_fingerprints':True,
            'physical_and_action_audits':'passed',
            'scope':'new cross-event ablation only; not historical grid provenance'}
    (RUN/'replay_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
