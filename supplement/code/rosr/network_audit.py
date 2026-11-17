"""Independent uplink accounting and public-plan audit, using capacity algebra."""
import math
from .boot import model, audit, EPS


def finish_by_capacity(uplink,vehicle,start,work):
    """Closed-form inverse cumulative service, independent of slot iteration."""
    period=uplink.vehicles*uplink.slot
    cycle=math.floor(start/period)
    served=cycle*uplink.slot+min(uplink.slot,max(0.,start-cycle*period-vehicle*uplink.slot))
    target=served+work/uplink.rate
    if work==0:return start
    last=max(0,math.ceil(target/uplink.slot-1e-10)-1)
    return last*period+vehicle*uplink.slot+target-last*uplink.slot


def check_episode(tasks,vehicles,multiplier,uplink,summary,rows,logs):
    audit.check_episode(summary,rows,logs)
    by_id={t.task_id:t for t in tasks};records={r['task_id']:r for r in rows}
    tails=[0.]*len(vehicles);positions=[(v.x,v.y) for v in vehicles]
    segments=[];committed=set();plans=0;last_routes=None;last_edge=None
    for e in logs:
        kind=e['kind'];now=e['time']
        if kind=='capture_complete':
            t=by_id[e['task_id']];positions[e['vehicle_id']]=(t.x,t.y)
        elif kind=='upload_commit':
            j=e['task_id'];vid=e['vehicle_id'];r=records[j]
            assert j not in committed
            expected=finish_by_capacity(uplink,vid,max(now,tails[vid]),r['upload'])
            assert abs(expected-e['finish'])<1e-7
            assert abs(r['capture_finish']-now)<EPS
            assert abs(r['upload_finish']-expected)<EPS
            total=0.
            for a,b in e['segments']:
                assert a>=max(now,tails[vid])-EPS and b>a
                cycle=math.floor((a-vid*uplink.slot+1e-10)/(uplink.vehicles*uplink.slot))
                low=cycle*uplink.vehicles*uplink.slot+vid*uplink.slot
                assert a>=low-EPS and b<=low+uplink.slot+EPS
                segments.append((a,b));total+=b-a
            assert abs(total*uplink.rate-r['upload'])<1e-7
            tails[vid]=expected;committed.add(j)
        elif kind=='plan':
            status={int(j):st for j,st in e['status_before'].items()}
            visible=set(e['visible_ids']);routes=tuple(tuple(r) for r in e['routes'])
            edge=tuple(e['edge_order']);assert len(routes)==len(vehicles)
            assert all(j in by_id and by_id[j].release<=now+EPS for j in visible)
            assert visible=={j for j,t in by_id.items() if t.release<=now+EPS}
            assert len(edge)==len(set(edge)) and len([j for r in routes for j in r])==len({j for r in routes for j in r})
            releases={j:max(now,records[j]['upload_finish']) for j,st in status.items() if st in ('ready','uploading')}
            assert all(abs(a-b)<EPS for a,b in zip(e['link_tails'],tails))
            captures={}
            for vid,route in enumerate(routes):
                active=e['active_vehicles'][str(vid)]
                cursor=now;loc=positions[vid];tail=max(now,tails[vid])
                if active is not None:
                    j=active['task_id'];t=by_id[j]
                    cursor=active['finish'];loc=(t.x,t.y)
                    assert records[j]['capture_start']<=now+EPS
                    assert abs(records[j]['capture_finish']-cursor)<EPS
                    if status[j]!='expired':
                        tail=finish_by_capacity(uplink,vid,max(cursor,tail),t.upload)
                        releases[j]=tail
                for j in route:
                    assert j in visible and status[j]=='pending'
                    t=by_id[j];v=vehicles[vid];assert t.sensor==v.sensor
                    cursor+=model.legacy.travel_time(*loc,t.x,t.y,v.speed)+t.service
                    tail=finish_by_capacity(uplink,vid,max(cursor,tail),t.upload)
                    captures[j]=cursor;releases[j]=tail;loc=(t.x,t.y)
            assert set(captures).issubset(edge)
            cursor=now if e['edge_active'] is None else e['edge_active'][2]
            for j in edge:
                assert j in releases
                start=max(cursor,releases[j]);end=start+by_id[j].work*multiplier
                assert end<=by_id[j].deadline+EPS
                a,b=e['edge_times'][str(j)]
                assert abs(a-start)<1e-7 and abs(b-end)<1e-7
                cursor=end
            assert abs(e['planned_value']-sum(by_id[j].value for j in edge))<EPS
            last_routes,last_edge=routes,edge;plans+=1
        elif kind=='capture_start':
            j=e['task_id'];vid=e['vehicle_id'];t=by_id[j]
            assert last_routes[vid][0]==j and t.release<=now+EPS
            travel=model.legacy.travel_time(*positions[vid],t.x,t.y,vehicles[vid].speed)
            assert abs(e['travel']-travel)<EPS and abs(e['finish']-(now+travel+t.service))<EPS
        elif kind=='edge_start':
            j=e['task_id'];assert last_edge[0]==j
            assert records[j]['upload_finish']<=now+EPS
            assert abs(e['finish']-(now+by_id[j].work*multiplier))<EPS
    segments.sort()
    assert all(a[1]<=b[0]+EPS for a,b in zip(segments,segments[1:]))
    assert all(t<=summary['end_time']+EPS for t in tails)
    assert committed=={r['task_id'] for r in rows if r.get('upload_finish') is not None}
    for r in rows:
        if r['edge_start'] is not None:
            assert r['upload_finish'] is not None and r['edge_start']>=r['upload_finish']-EPS
    return dict(plans=plans,packets=len(committed),segments=len(segments))
