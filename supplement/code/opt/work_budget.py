"""Diagnostic work ceilings for frozen policies, without changing code/ files.

Module-local budget clocks disable ordinary wall gates, not Python's clock or
the simulation clock. MIP nodes/calls and LP iterations are bounded instead.
A real wall guard remains insurance; any hit invalidates a reproducibility
claim for that episode. This mode is not the original 0.1-second algorithm.
"""
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import time
import warnings

import numpy as np
from scipy.optimize import Bounds,LinearConstraint,OptimizeResult
from scipy.sparse import coo_array

from rosr import core,capacity
from rosr import policy as policy_module

BASE_BUILDER=core._Builder


@dataclass(frozen=True)
class WorkLimits:
    mip_nodes:int=256
    mip_calls:int=16
    lp_iterations:int=10000
    wall_guard_seconds:float=5.

    def __post_init__(self):
        if (not isinstance(self.mip_nodes,int) or self.mip_nodes<1
                or not isinstance(self.mip_calls,int) or not 1<=self.mip_calls<=16
                or not isinstance(self.lp_iterations,int) or self.lp_iterations<1
                or not np.isfinite(self.wall_guard_seconds) or self.wall_guard_seconds<=0):
            raise ValueError('positive node/iteration limits, 1..16 MIP calls, and a finite wall guard required')


class DecisionWork:
    def __init__(self,limits,now=time.perf_counter):
        self.limits=limits
        self.now=now
        self.started=now()
        self.used=Counter()
        self.guard_hit=False

    @property
    def nodes_left(self):
        return max(0,self.limits.mip_nodes-self.used['mip_nodes'])

    @property
    def exhausted(self):
        return self.nodes_left==0 or self.used['mip_calls']>=self.limits.mip_calls

    def remaining_wall(self):
        remaining=self.limits.wall_guard_seconds-(self.now()-self.started)
        if remaining<=0:
            self.guard_hit=True
        return max(0.,remaining)


class BudgetClock:
    """A module-local stopping signal. Real timings are measured by the adapter."""
    def __init__(self,work):
        self.work=work

    def perf_counter(self):
        return 1e6 if self.work.remaining_wall()<=0 or self.work.exhausted else 0.


class NodeBuilder(BASE_BUILDER):
    def __init__(self,work):
        super().__init__()
        self.work=work

    def solve(self,seconds):
        work=self.work
        remaining=work.remaining_wall()
        if work.exhausted or remaining<=0:
            return OptimizeResult(status=1,x=None,fun=None,mip_dual_bound=None,
                message='Diagnostic work ceiling or wall insurance reached')
        matrix=coo_array((np.asarray(self.data,dtype=float),
            (np.asarray(self.rows,dtype=np.int32),np.asarray(self.cols,dtype=np.int32))),
            shape=(len(self.lo),len(self.keys))).tocsc()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='Unrecognized options detected.*',category=RuntimeWarning)
            result=core.milp(np.asarray(self.objective,dtype=float),
                integrality=np.asarray(self.integer,dtype=np.int32),
                bounds=Bounds(np.zeros(len(self.keys)),np.asarray(self.ub,dtype=float)),
                constraints=LinearConstraint(matrix,np.asarray(self.lo),np.asarray(self.hi)),
                options={'presolve':True,'mip_rel_gap':0.,'threads':1,
                         'node_limit':work.nodes_left,'time_limit':max(.001,remaining)})
        return result


def time_stopped(result):
    return getattr(result,'status',None)==1 and 'time limit' in str(getattr(result,'message','')).lower()


@contextmanager
def work_bindings(work,search_control=True,lp_control=True):
    """Single sequential decision in one process; restore every binding on exit."""
    old_policy_time,old_core_time=policy_module.time,core.time
    old_builder,old_milp,old_lp=core._Builder,core.milp,capacity.linprog

    def observed_mip(*args,**kwargs):
        requested=kwargs.get('options',{}).get('node_limit')
        work.used['mip_calls']+=1
        result=old_milp(*args,**kwargs)
        nodes=getattr(result,'mip_node_count',None)
        if nodes is not None and np.isfinite(nodes):
            work.used['mip_nodes']+=int(nodes)
            if requested is not None and nodes>requested:
                work.used['node_limit_overshoots']+=1
        elif getattr(result,'status',None)==1 and search_control:
            # No reported work count on a limited solve: charge its entire
            # allowance rather than silently giving the next solve more work.
            work.used['mip_nodes']+=int(requested or 0)
            work.used['unreported_limited_node_counts']+=1
        if getattr(result,'status',None)==1:
            work.used['mip_limit_stops']+=1
        if time_stopped(result):
            work.used['mip_time_stops']+=1
            if search_control:
                work.guard_hit=True
        return result

    def observed_lp(*args,**kwargs):
        options=dict(kwargs.get('options') or {})
        if lp_control:
            options.update(maxiter=work.limits.lp_iterations,
                           time_limit=max(.001,work.remaining_wall()),threads=1)
            kwargs=dict(kwargs,options=options)
        work.used['lp_calls']+=1
        result=old_lp(*args,**kwargs)
        iterations=getattr(result,'nit',0)
        if iterations is not None:
            work.used['lp_iterations']+=int(iterations)
        if getattr(result,'status',None)==1:
            work.used['lp_limit_stops']+=1
        if time_stopped(result):
            work.used['lp_time_stops']+=1
            if lp_control:
                work.guard_hit=True
        return result

    core.milp=observed_mip
    capacity.linprog=observed_lp
    if search_control:
        clock=BudgetClock(work)
        policy_module.time,core.time=clock,clock
        core._Builder=lambda:NodeBuilder(work)
    try:
        yield
    finally:
        policy_module.time,core.time=old_policy_time,old_core_time
        core._Builder,core.milp,capacity.linprog=old_builder,old_milp,old_lp


class WorkBudgetPolicy:
    """Wrap RCR or its pool candidates for work/LP-only/search-only diagnostics."""
    def __init__(self,inner,limits=None,mode='work'):
        if mode not in ('work','lp_only','search_only'):
            raise ValueError('expected work, lp_only, or search_only diagnostic mode')
        self.inner=inner
        self.limits=limits or WorkLimits()
        self.mode=mode
        self.decisions=[]
        self.optimization_stats=Counter()
        self.total_wall_seconds=0.

    @property
    def stats(self):
        values=dict(vars(self.inner.stats))
        # Clock-disabled legacy timing fields are not real measurements.
        values.pop('initialization_seconds',None)
        values.pop('search_seconds',None)
        values['planning_seconds']=self.total_wall_seconds
        return values

    def __call__(self,s):
        started=time.perf_counter()
        work=DecisionWork(self.limits)
        search=self.mode in ('work','search_only')
        lp=self.mode in ('work','lp_only')
        with work_bindings(work,search_control=search,lp_control=lp):
            plan=self.inner(s)
        work.remaining_wall()
        self.optimization_stats.update(work.used)
        self.optimization_stats['wall_guard_hits']+=int(work.guard_hit)
        self.optimization_stats['work_ceiling_events']+=int(search and work.exhausted)
        entry=dict(self.inner.decisions[-1])
        meta=dict(entry.get('meta',{}))
        if search:
            meta['elapsed_seconds']=None
            meta['solver_budget_seconds']=None
            meta['legacy_stop_reason']=meta.get('stop_reason')
            if work.guard_hit:
                meta['stop_reason']='wall_guard_insurance'
            elif work.exhausted:
                meta['stop_reason']='diagnostic_work_ceiling'
            elif meta.get('stop_reason')=='time_limited_value_solution':
                meta['stop_reason']='node_limited_value_solution'
            entry['initialization_ms']=None
        meta['diagnostic_budget_mode']=self.mode
        entry['meta']=meta
        entry['work_used']=dict(work.used)
        entry['wall_guard_hit']=work.guard_hit
        entry['work_exhausted']=search and work.exhausted
        elapsed=time.perf_counter()-started
        entry['planning_ms']=1000*elapsed
        self.total_wall_seconds+=elapsed
        self.decisions.append(entry)
        return plan
