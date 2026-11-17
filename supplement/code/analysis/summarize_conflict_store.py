"""Summarize the audited cross-event conflict-memory experiment.

Run from the project root:
    python -B code/analysis/summarize_conflict_store.py
The paired Student intervals are descriptive; wall-clock times are susceptible
to worker contention and timeout-dependent trajectory variation.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'opt/runs/eswa_certificate_confirmation_20261112'
OUT=Path(__file__).resolve().parent
CONDITIONS=('coupled_primary','edge_dominant','tighter_deadlines')
VARIANTS=('plain_single_event','plain_cross_event','strengthened_cross_event')


def interval(values):
    m=mean(values);h=float(t.ppf(.975,len(values)-1))*stdev(values)/math.sqrt(len(values))
    return m,m-h,m+h


def main():
    data={}
    for condition in CONDITIONS:
        for seed in range(20423000,20423030):
            for variant in VARIANTS:
                path=RUN/condition/str(seed)/variant/'summary.json'
                row=json.loads(path.read_text(encoding='utf8'))
                assert row['audited'] and (row['profile'],row['seed'],row['variant'])==(condition,seed,variant)
                data[condition,seed,variant]=row
            assert len({data[condition,seed,v]['stream_hash'] for v in VARIANTS})==1
    assert len(data)==270
    cells=[]; contrasts=[]
    for condition in CONDITIONS:
        baseline=[data[condition,seed,VARIANTS[0]] for seed in range(20423000,20423030)]
        for variant in VARIANTS:
            rows=[data[condition,seed,variant] for seed in range(20423000,20423030)]
            differences=[100*(a['value_rate']-b['value_rate']) for a,b in zip(rows,baseline)]
            dm,lo,hi=interval(differences)
            ms=mean(r['planning_ms'] for r in rows)/1000
            cell=dict(condition=condition,variant=variant,n=30,
                      mean_value_percent=100*mean(r['value_rate'] for r in rows),
                      mean_planning_seconds=ms,
                      delta_value_vs_single_pp=dm,delta_low95_pp=lo,delta_high95_pp=hi,
                      wins=sum(x>1e-9 for x in differences),ties=sum(abs(x)<=1e-9 for x in differences),
                      losses=sum(x< -1e-9 for x in differences),
                      mean_reused_conflict_cuts=mean(r['policy_stats']['reused_conflict_cuts'] for r in rows),
                      mean_entries_discarded=mean(r['entries_discarded'] or 0 for r in rows))
            cells.append(cell)
            print(condition,variant,'value',f"{cell['mean_value_percent']:.3f}",'time',f'{ms:.3f}',
                  'delta',f'{dm:+.3f}',f'[{lo:+.3f},{hi:+.3f}]',
                  'W/T/L',f"{cell['wins']}/{cell['ties']}/{cell['losses']}",
                  'reuse',f"{cell['mean_reused_conflict_cuts']:.1f}",
                  'discarded',f"{cell['mean_entries_discarded']:.1f}")
        for right,left in [(VARIANTS[1],VARIANTS[0]),(VARIANTS[2],VARIANTS[1])]:
            rs=[data[condition,seed,right] for seed in range(20423000,20423030)]
            ls=[data[condition,seed,left] for seed in range(20423000,20423030)]
            values=[100*(a['value_rate']-b['value_rate']) for a,b in zip(rs,ls)]
            times=[(a['planning_ms']-b['planning_ms'])/1000 for a,b in zip(rs,ls)]
            vm,vl,vh=interval(values);tm,tl,th=interval(times)
            ratio=(mean(a['planning_ms'] for a in rs)/mean(b['planning_ms'] for b in ls)-1)*100
            contrasts.append(dict(condition=condition,variant=right,control=left,n=30,
                                  delta_value_pp=vm,value_low95_pp=vl,value_high95_pp=vh,
                                  delta_time_s=tm,time_low95_s=tl,time_high95_s=th,
                                  ratio_of_mean_times_pct=ratio))
            print('paired contrast',condition,right,'vs',left,'time',f'{tm:+.3f}',f'[{tl:+.3f},{th:+.3f}]',
                  'ratio',f'{ratio:+.1f}%', 'value',f'{vm:+.3f}',f'[{vl:+.3f},{vh:+.3f}]')
    for name,rows in [('conflict_reuse_cells.csv',cells),('conflict_reuse_contrasts.csv',contrasts)]:
        with (OUT/name).open('w',encoding='utf8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Recorded',len(data),'audited, seed-matched episode summaries.')


if __name__=='__main__': main()
