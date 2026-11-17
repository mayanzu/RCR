"""Post hoc alternative multiplicity families and Hefei signed-rank checks.

Run: python -B paper/analysis/reanalysis_sensitivity.py

The grid-family split is a sensitivity analysis chosen after results were seen;
it does not control family-wise error jointly across the two split families.
Hefei paired differences use exact rational completed/total values rather than
floating-point differences, preserving true ties. Raw and Holm-adjusted p
values are included for all 12 road comparisons; the 3 Density comparisons are
also adjusted as a separate, post hoc family.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from decimal import Decimal
from fractions import Fraction
from pathlib import Path


ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
GRID=OUT/'reference_signed_rank.csv'
ROAD=ROOT/'opt/osm_hefei/results/confirmation/episodes.csv'
CONDITIONS=('coupled_primary','edge_dominant','tighter_deadlines')
CONTROLS=('e2e_resource_density','joint_lns','coalition_value','lit_dpso')


def exact_rank_test(differences: list[Fraction]) -> tuple[float,float,int]:
    nonzero=[x for x in differences if x]
    if not nonzero:
        return 0.,1.,0
    # Twice the average rank is integer even when ranks are tied.
    counts=Counter(abs(x) for x in nonzero)
    rank2={}
    position=1
    for magnitude,count in sorted(counts.items()):
        rank2[magnitude]=2*position+count-1
        position+=count
    ranks=[rank2[abs(x)] for x in nonzero]
    positive=sum(r for r,x in zip(ranks,nonzero) if x>0)
    observed=min(positive,sum(ranks)-positive)
    probabilities=Counter({0:1})
    for rank in ranks:
        updated=probabilities.copy()
        for total,n in probabilities.items():
            updated[total+rank]+=n
        probabilities=updated
    p=min(1.,2*sum(n for total,n in probabilities.items() if total<=observed)/2**len(nonzero))
    return observed/2,p,len(nonzero)


def holm(rows: list[dict], output_key: str) -> None:
    previous=0.
    for i,row in enumerate(sorted(rows,key=lambda x:float(x['p_unadjusted']))):
        previous=max(previous,min(1.,(len(rows)-i)*float(row['p_unadjusted'])))
        row[output_key]=previous


def write(path:Path, rows:list[dict]) -> None:
    with path.open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main() -> None:
    with GRID.open(encoding='utf8',newline='') as f:
        grid=list(csv.DictReader(f))
    assert len(grid)==40
    for r in grid:
        r['p_holm_40']=float(r.pop('p_holm'))
        r['p_holm_split']=0.
        r['family']='internal_10' if r['control'] in CONTROLS[:2] else 'external_30'
    for family in ('internal_10','external_30'):
        subset=[r for r in grid if r['family']==family]
        assert len(subset)==(10 if family=='internal_10' else 30)
        holm(subset,'p_holm_split')
    write(OUT/'reference_family_sensitivity.csv',grid)

    with ROAD.open(encoding='utf8',newline='') as f:
        source=list(csv.DictReader(f))
    data={}
    for r in source:
        key=(r['profile'],r['method'],int(r['seed']))
        if key in data: raise ValueError(f'duplicate {key}')
        data[key]=r
    assert len(data)==450
    road=[]
    for condition in CONDITIONS:
        for control in CONTROLS:
            ds=[]
            for seed in range(20480000,20480030):
                a,b=data[(condition,'rosr',seed)],data[(condition,control,seed)]
                assert a['stream_hash']==b['stream_hash']
                ta=Fraction(Decimal(a['total_value']));tb=Fraction(Decimal(b['total_value']))
                assert ta==tb and ta>0
                va=Fraction(Decimal(a['completed_value']));vb=Fraction(Decimal(b['completed_value']))
                ds.append(100*(va-vb)/ta)
            w,p,n=exact_rank_test(ds)
            road.append(dict(condition=condition,control=control,n_paired=len(ds),n_nonzero=n,
                             wins=sum(x>0 for x in ds),ties=sum(x==0 for x in ds),losses=sum(x<0 for x in ds),
                             mean_difference_pp=float(sum(ds)/len(ds)),signed_rank_w=w,
                             p_unadjusted=p,p_holm_12=0.,p_holm_density3=''))
    holm(road,'p_holm_12')
    holm([r for r in road if r['control']=='e2e_resource_density'],'p_holm_density3')
    write(OUT/'hefei_signed_rank.csv',road)
    (OUT/'sensitivity_metadata.json').write_text(json.dumps(dict(
        post_hoc=True,reference_original_family=40,reference_split=[10,30],
        hefei_full_family=12,hefei_density_sensitivity_family=3,
        cross_family_error_control='not claimed',hefei_differences='exact rational completed_value / total_value',
        grid_csv_sha256=hashlib.sha256(GRID.read_bytes()).hexdigest(),
        hefei_csv_sha256=hashlib.sha256(ROAD.read_bytes()).hexdigest()),indent=2)+'\n',encoding='utf8')
    for r in grid:
        if r['control'] in CONTROLS[:2] and r['condition'] in CONDITIONS:
            print('grid',r['condition'],r['control'],'p40',f"{float(r['p_holm_40']):.6g}",
                  'p10',f"{float(r['p_holm_split']):.6g}")
    for r in road:
        print('road',r['condition'],r['control'],'W/T/L',f"{r['wins']}/{r['ties']}/{r['losses']}",
              'raw',f"{r['p_unadjusted']:.6g}",'p12',f"{r['p_holm_12']:.6g}",
              'p3',f"{r['p_holm_density3']:.6g}" if r['p_holm_density3']!='' else '-')


if __name__=='__main__':
    main()
