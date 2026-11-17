"""Complete factorial service-model comparison with independent uplink audit."""

from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[1]/'code'))
from bootstrap import BASELINES_ROOT,output_directory,source_hash
import argparse,json,time,os,platform
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from run_rosr_bench import save,parse_seeds
from rosr import RosrPolicy,model,audit,verify_frozen
from rosr.comparison import DensityPolicy,JointLnsPolicy
from rosr.network_model import NetworkEnv,SCENARIOS,generate
from rosr.network_audit import check_episode
from baselines.value_coalition import ValueCoalitionPolicy
import hashlib

METHODS=('e2e_resource_density','rosr','joint_lns')
EXTERNAL=('edge_adapter','e2e_repack','vehicle_repack_edf','coalition_value')


def worker(job):
    seed,name,n,output,seconds=job
    sc=next(s for s in SCENARIOS if s.name==name)
    out=Path(output)/str(seed)/name
    # One worker executes all methods for the same instance, retaining common
    # exogenous inputs. Each method receives a fresh environment and state.
    tasks,vehicles=generate(seed,sc,n_vehicles=n)
    results=[]
    methods=METHODS if sc.network else METHODS+EXTERNAL
    shift=(seed+SCENARIOS.index(sc))%len(methods)
    for method in methods[shift:]+methods[:shift]:
        ep=out/method
        provenance=dict(source_sha256=source_hash(),scenario=asdict(sc),n_vehicles=n,seconds=seconds)
        if method=='coalition_value':provenance['control_sha256']=hashlib.sha256((BASELINES_ROOT/'value_coalition.py').read_bytes()).hexdigest()
        if (ep/'summary.json').exists():
            assert json.loads((ep/'provenance.json').read_text())==provenance,'stale model cache'
            results.append(json.loads((ep/'summary.json').read_text()));continue
        env=NetworkEnv(tasks,vehicles,1.5) if sc.network else model.Env(tasks,vehicles,1.5)
        factory={'rosr':RosrPolicy,'e2e_resource_density':DensityPolicy,'joint_lns':JointLnsPolicy,'coalition_value':ValueCoalitionPolicy}.get(method)
        policy=factory(seconds=seconds,seed=seed) if factory else None
        while not env.done:
            started=time.perf_counter();p=policy(env.snapshot()) if policy else model.make_plan(env.snapshot(),method)
            env.planning_seconds+=time.perf_counter()-started
            env.apply(p);env.advance()
            assert env.event_index<10000
        summary,rows,logs=env.results()
        summary.update(seed=seed,scenario=name,method=method,n_vehicles=n,
                       stream_hash=model.signature(tasks,vehicles))
        if sc.network:
            summary['network_audit']=check_episode(tasks,vehicles,1.5,env.uplink,summary,rows,logs)
        else:
            audit.check_episode(summary,rows,logs)
            audit.audit_action_logs(tasks,vehicles,1.5,rows,logs)
        save(ep/'provenance.json',provenance);save(ep/'summary.json',summary)
        save(ep/'tasks.json',rows)
        if policy:save(ep/'decisions.json',policy.decisions)
        (ep/'events.jsonl').write_text('\n'.join(json.dumps(e) for e in logs)+'\n',encoding='utf-8')
        results.append(summary)
        print(seed,name,method,f'J={100*summary["value_rate"]:.3f}',
              f'cost={summary["planning_ms"]/1000:.2f}s',flush=True)
    return results


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',default='20362000-20362029')
    ap.add_argument('--scenarios',default=','.join(s.name for s in SCENARIOS))
    ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--vehicles',type=int,default=8)
    ap.add_argument('--seconds',type=float,default=.1)
    ap.add_argument('--output',default=None,help='Directory outside this package')
    ap.add_argument('--label',default='expanded_models')
    args=ap.parse_args();verify_frozen()
    out=output_directory(args.output,label=args.label)
    save(out/'settings.json',dict(vars(args),source_sha256=source_hash(),python=platform.python_version(),
        platform=platform.platform(),processor=platform.processor(),cpu_count=os.cpu_count(),
        thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}))
    jobs=[(seed,sc,args.vehicles,str(out),args.seconds) for seed in parse_seeds(args.seeds) for sc in args.scenarios.split(',')]
    with ProcessPoolExecutor(args.workers) as pool:
        for f in as_completed([pool.submit(worker,j) for j in jobs]):f.result()


if __name__=='__main__':main()
