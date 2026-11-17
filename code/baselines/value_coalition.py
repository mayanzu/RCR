"""Author-adapted value refinement of the frozen coalition route initializer.

This is a stronger comparison control, not a reproduction of a published
value-aware algorithm. It keeps the original initializer, then tests route
transfer/insertion and one-for-one acquisition replacement using actual
completed-report value under the fixed-upload contract.
"""
import time
from rosr import model
from rosr.boot import EPS
from rosr.evaluation import EvaluationContext
from rosr.policy import independent_value_bound


class ValueCoalitionPolicy:
    def __init__(self,seconds=.1,max_evaluations=1024,**kwargs):
        self.seconds,self.max_evaluations=seconds,max_evaluations
        self.decisions=[];self.stats={}

    def __call__(self,s):
        if getattr(s,'uplink',None) is not None:
            raise ValueError('Coalition-V is defined for the fixed-upload contract only')
        start=time.perf_counter()
        base=model.vehicle_plan(s,'edge_adapter')
        initialized=time.perf_counter();deadline=initialized+self.seconds
        ctx=EvaluationContext(s);best=base;evaluations=0
        # Accepted transfers cannot return a task to an already visited vehicle.
        history={j:{i} for i,r in enumerate(base.routes) for j in r}
        def stopped():return evaluations>=self.max_evaluations or time.perf_counter()>=deadline
        if independent_value_bound(s,ctx)>base.value+EPS:
            changed=True
            while changed and not stopped():
                changed=False
                for j in sorted((j for j,st in s.status.items() if st=='pending'),
                                key=lambda j:(-s.tasks[j].value,s.tasks[j].deadline,j)):
                    current=best
                    source=next((i for i,r in enumerate(current.routes) if j in r),None)
                    drops=[None] if source is not None else [None,*sorted(current.capture_times,key=lambda h:(s.tasks[h].value,h))]
                    for dropped in drops:
                        routes=[list(r) for r in current.routes]
                        if source is not None:routes[source].remove(j)
                        if dropped is not None:
                            for r in routes:
                                if dropped in r:r.remove(dropped)
                        for vid,v in enumerate(s.vehicles):
                            if v.sensor!=s.tasks[j].sensor or (vid in history.get(j,set()) and vid!=source):continue
                            for pos in range(len(routes[vid])+1):
                                if stopped():break
                                candidate=[r[:] for r in routes];candidate[vid].insert(pos,j)
                                evaluations+=1
                                # Fairness (N2): every acquisition in the trial
                                # routes must have a feasible processing slot,
                                # matching the action space of RCR/Density/Joint
                                # DR. Optional ready reports stay selectable.
                                p=ctx.evaluate(candidate,required=set(
                                    j2 for r in candidate for j2 in r))
                                if p is not None and p.value>best.value+EPS:
                                    best=p;history.setdefault(j,set()).add(vid);changed=True
                            if stopped():break
                        if stopped():break
                    if stopped():break
        self.decisions.append(dict(event=s.event_index,mode='coalition_value_gain' if best.value>base.value+EPS else 'keep_coalition',
            baseline_value=base.value,chosen_value=best.value,evaluations=evaluations,
            initialization_ms=1000*(initialized-start),planning_ms=1000*(time.perf_counter()-start),
            stopped_by='budget' if time.perf_counter()>=deadline else 'work_limit' if evaluations>=self.max_evaluations else 'neighborhood_exhausted'))
        return best
