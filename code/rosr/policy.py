"""Density initialization with release-conflict route reoptimization."""
from __future__ import annotations
import time
from dataclasses import dataclass
from .boot import EPS, model, validate_plan, action_key
from .core import enumerate_routes, solve_set_packing
from .evaluation import EvaluationContext, density_initialize, local_equal_value_probe

from .criteria import (acceptance_reason, plan_resources, plan_min_slack,
                       sunk_value, SLACK_GATE, ACCEPTED, exposure_cost, release_profile)
from .continuation import Continuation
from .capacity import deadline_value_bound, release_lower_bounds, release_value_bound
from .edge import ConflictStore

ALGORITHM_VERSION = 'release_conflict_continuation_2026-10-25'
DEFAULT_BETA = 0.25


def independent_value_bound(s, ctx):
    return deadline_value_bound(s, release_lower_bounds(s, ctx))

@dataclass
class PolicyStats:
    events:int=0
    opportunities:int=0
    actions_changed:int=0
    strict_improvements:int=0
    compact_ties:int=0
    planning_seconds:float=0.
    initialization_seconds:float=0.
    search_seconds:float=0.
    routes_enumerated:int=0
    horn_cuts:int=0
    no_good_cuts:int=0
    analytic_skips:int=0
    initialization_bound_stops:int=0
    energy_rejects:int=0
    certificate_rejects:int=0
    reused_conflict_cuts:int=0
    release_bound_improvements:int=0
    continuation_kept:int=0
    continuation_invalidations:int=0
    same_action_tail_gains:int=0
    value_bound_tie_searches:int=0
    action_rejections:int=0
    density_rebuild_skips:int=0
    tie_probe_hits:int=0

class RosrPolicy:
    def __init__(self,beta=DEFAULT_BETA,max_depth=3,seconds=0.1,seed=0,
                 compact_ties=True,conflict_cuts=True,urgency_alpha=0.,slack_price=0.,
                 pool_routes=64,pool_factor=4):
        if beta<0 or max_depth<1 or seconds<=0:
            raise ValueError('beta >= 0, max_depth >= 1, seconds > 0 required')
        if pool_routes<1 or pool_factor<1:
            raise ValueError('pool_routes >= 1, pool_factor >= 1 required')
        self.beta,self.max_depth,self.seconds,self.seed=beta,max_depth,seconds,seed
        self.compact_ties,self.conflict_cuts=compact_ties,conflict_cuts
        self.urgency_alpha,self.slack_price=urgency_alpha,slack_price
        self.pool_routes,self.pool_factor=pool_routes,pool_factor
        self.stats,self.decisions=PolicyStats(),[]
        self.conflicts = ConflictStore()
        self.last_now = -float('inf')
        self.last_event = -1
        self.continuation = Continuation()

    def __call__(self,s):
        started=time.perf_counter()
        if s.now < self.last_now or s.event_index < self.last_event:
            self.conflicts = ConflictStore()
            self.continuation = Continuation()
        self.last_now = s.now
        self.last_event = s.event_index
        ctx=EvaluationContext(s, conflicts=self.conflicts if self.conflict_cuts else None)
        carried, carry_status = self.continuation.advance(s)
        previous_guaranteed = self.continuation.guaranteed
        bound_started=time.perf_counter()
        releases=release_lower_bounds(s,ctx)
        bound=deadline_value_bound(s,releases)
        bound_seconds=time.perf_counter()-bound_started
        st=self.stats
        capture_possible=any(v.active is None and any(
            status=='pending' and s.tasks[j].sensor==v.sensor
            for j,status in s.status.items()) for v in s.vehicles)
        edge_possible=s.edge_active is None and 'ready' in s.status.values()
        # At the certified value ceiling a full density rebuild cannot raise
        # value. Strict-value mode simply keeps the incumbent; compact-ties
        # mode still looks for another action via a bounded local probe, so
        # value-ceiling events never unconditionally abandon other actions.
        ceiling = carried is not None and bound <= carried.value + EPS
        density_rebuild_skipped = False
        if ceiling and (not self.compact_ties or not (capture_possible or edge_possible)):
            density, density_rebuild_skipped = carried, True
        elif ceiling:
            density = local_equal_value_probe(s, ctx, carried)
            density_rebuild_skipped = True
            st.tie_probe_hits += int(density is not None)
            density = carried if density is None else density
        else:
            # Full rebuild from empty. A warm start seeded with the incumbent
            # routes was tried and reverted: it locks in carried acquisitions,
            # loses drop-and-reconfigure moves, and cost up to -2.7 pp on the
            # streams where the full rebuild found the largest gains, with no
            # reliable speedup (see 算法性能优化方案.md §8.5).
            density=density_initialize(s,ctx,value_upper=bound,
                                       urgency_alpha=self.urgency_alpha)
        # A valid incumbent wins ties; replacements obey the same guard as
        # search candidates. Density is a proposal, not a fresh guarantee.
        baseline = carried if carried is not None else density
        if carried is not None and density is not carried and acceptance_reason(s, density, carried,
                compact_ties=self.compact_ties, guarded=True, slack_price=self.slack_price) in ACCEPTED:
            baseline = density
        initialized=time.perf_counter()
        initialization_seconds=initialized-started-bound_seconds
        deadline=initialized+self.seconds-bound_seconds
        st.events+=1
        st.density_rebuild_skips+=int(density_rebuild_skipped)
        st.continuation_kept+=int(carried is not None and baseline is carried)
        st.continuation_invalidations+=int(carry_status not in ('valid','empty'))
        st.initialization_seconds+=initialization_seconds
        proposed,meta=None,{}
        cheap_bound=bound
        if (capture_possible or edge_possible) and bound>baseline.value+EPS:
            bound=release_value_bound(s,releases,bound,
                seconds=min(.005,self.seconds/10,max(0.,deadline-time.perf_counter())))
            st.release_bound_improvements+=int(bound<cheap_bound-EPS)
        # At the value ceiling, a different known density action is cheap
        # evidence that a resource-only search can change immediate behavior.
        tie_search = (self.compact_ties and bound<=baseline.value+EPS
                      and density.value>=baseline.value-EPS
                      and action_key(s,density)!=action_key(s,baseline))
        st.value_bound_tie_searches+=int(tie_search)
        if (capture_possible or edge_possible) and (bound>baseline.value+EPS or tie_search):
            st.opportunities+=1
            focus=set().union(*(set(floors) for floors in self.conflicts.active(s))) if self.conflict_cuts else set()
            pool=[enumerate_routes(s,vid,max_depth=self.max_depth,anchor=baseline.routes[vid],
                                   focus=focus,stop=lambda:time.perf_counter()>=deadline,
                                   max_routes=self.pool_routes,pool_factor=self.pool_factor)
                  for vid in range(len(s.vehicles))]
            remaining=deadline-time.perf_counter()
            if remaining>0:
                _,_,meta=solve_set_packing(s,pool,ctx.base_reports,beta=self.beta,
                    seconds=remaining,baseline=baseline,strict_value_only=not self.compact_ties,
                    conflict_cuts=self.conflict_cuts,
                    conflict_store=self.conflicts if self.conflict_cuts else None)
                proposed=meta.pop('plan',None)
            else:
                meta=dict(stop_reason='pool_budget',routes_enumerated=sum(map(len,pool)))
        else:
            st.analytic_skips+=1
            meta['stop_reason']='independent_value_bound' if capture_possible or edge_possible else 'no_action'
            meta['bound_kind']='fractional_prefix_and_dual_certified_release_energy'
        reason=acceptance_reason(s,proposed,baseline,compact_ties=self.compact_ties,guarded=True,
                                 slack_price=self.slack_price)
        chosen=proposed if reason in ACCEPTED else baseline
        changed=action_key(s,chosen)!=action_key(s,density)
        st.actions_changed+=int(changed)
        st.strict_improvements+=int(reason=='strict_value_gain')
        st.compact_ties+=int(reason=='equal_value_resource_dominance')
        st.same_action_tail_gains+=int(reason=='same_action_tail_gain')
        st.action_rejections+=meta.get('action_rejections',0)
        validate_plan(s,chosen)
        assert chosen.value>=baseline.value-EPS
        guaranteed = Continuation.value(s,chosen)
        if carried is not None and previous_guaranteed is not None:
            assert guaranteed>=previous_guaranteed-EPS, 'continuation value was lost'
        self.continuation.remember(s,chosen)
        st.routes_enumerated+=meta.get('routes_enumerated',0)
        st.horn_cuts+=meta.get('horn_cuts',0)
        st.no_good_cuts+=meta.get('no_good_cuts',0)
        st.initialization_bound_stops+=ctx.bound_stops
        st.energy_rejects+=ctx.energy_rejects
        st.certificate_rejects+=ctx.certificate_rejects
        st.reused_conflict_cuts+=meta.get('reused_conflict_cuts',0)
        # Profiles are logged per decision; compute once and share with cost.
        baseline_profile=release_profile(s,baseline)
        chosen_profile=release_profile(s,chosen)
        elapsed=time.perf_counter()-started
        st.planning_seconds+=elapsed
        st.search_seconds+=elapsed-initialization_seconds
        self.decisions.append(dict(event=s.event_index,time=s.now,seed=self.seed,
            version=ALGORITHM_VERSION,mode=reason,baseline_value=baseline.value,
            chosen_value=chosen.value,proposed_value=None if proposed is None else proposed.value,
            density_value=density.value,continuation_status=carry_status,
            continuation_value=None if carried is None else carried.value,
            guaranteed_value=guaranteed,previous_guaranteed_value=previous_guaranteed,
            exposure_cost=exposure_cost(s,chosen,baseline,baseline_profile,chosen_profile),
            release_profile=chosen_profile,
            independent_upper=bound,cheap_upper=cheap_bound,
            initialization_bound_stops=ctx.bound_stops,energy_rejects=ctx.energy_rejects,
            certificate_rejects=ctx.certificate_rejects,baseline_resources=plan_resources(s,baseline),
            chosen_resources=plan_resources(s,chosen),changed=changed,meta=meta,
            initialization_ms=1000*initialization_seconds,planning_ms=1000*elapsed,
            evaluation_calls=ctx.calls,evaluation_cache_hits=ctx.hits,
            baseline_routes=baseline.routes,chosen_routes=chosen.routes,
            baseline_edge_order=baseline.edge_order,chosen_edge_order=chosen.edge_order))
        return chosen

