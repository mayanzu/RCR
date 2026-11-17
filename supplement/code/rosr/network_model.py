"""Auditable reserved-uplink extension of the immutable event service model."""
from dataclasses import asdict, dataclass, replace
import copy
from .boot import model, EPS, report_releases, validate_plan
from .transmission import Uplink


@dataclass
class NetworkSnapshot(model.Snapshot):
    uplink: Uplink | None = None
    link_tails: tuple = ()


@dataclass(frozen=True)
class Scenario:
    name: str
    batch: bool = False
    correlated: bool = False
    network: bool = False


SCENARIOS = tuple(Scenario(f'b{b}_c{c}_u{u}',bool(b),bool(c),bool(u))
                  for b in (0,1) for c in (0,1) for u in (0,1))


def generate(seed,scenario,n_vehicles=8):
    tasks,vehicles=model.generate(seed,model.Profile('coupled_primary'),n_vehicles=n_vehicles)
    if scenario.batch:
        tasks=[replace(t,release=(t.release//5)*5,
                       deadline=t.deadline-t.release+(t.release//5)*5) for t in tasks]
    if scenario.correlated:
        ranked=sorted(tasks,key=lambda t:(t.service+1.5*t.work,t.task_id))
        priorities=sorted(t.value for t in tasks)
        values={t.task_id:v for t,v in zip(ranked,priorities)}
        tasks=[replace(t,value=values[t.task_id]) for t in tasks]
    return tasks,vehicles


class NetworkEnv(model.Env):
    def __init__(self,tasks,vehicles,multiplier,uplink=None):
        self.uplink=uplink or Uplink(len(vehicles))
        if self.uplink.vehicles!=len(vehicles):
            raise ValueError('calendar and fleet must agree')
        self.link_tails=[0.]*len(vehicles)
        self.uploads={}
        super().__init__(tasks,vehicles,multiplier)

    def snapshot(self):
        return NetworkSnapshot(self.now,self.known.copy(),self.status.copy(),
            model.copy_vehicles_for_snapshot(self.vehicles),self.ready_times.copy(),self.edge_active,
            self.multiplier,self.event_index,self.uplink,tuple(self.link_tails))

    def _settle(self):
        before=len(self.logs)
        super()._settle()
        # Completed captures at this timestamp enter their own link FIFO.
        for e in self.logs[before:]:
            if e['kind']!='capture_complete':
                continue
            j,vid=e['task_id'],e['vehicle_id']
            if self.status[j]=='expired':
                continue
            t=self.known[j]
            finish,segments=self.uplink.transmit(vid,max(self.now,self.link_tails[vid]),t.upload)
            self.link_tails[vid]=finish
            self.ready_times[j]=finish
            self.records[j]['upload_segments']=segments
            self.records[j]['upload_finish']=finish
            self.uploads[j]=finish
            self.log('upload_commit',task_id=j,vehicle_id=vid,finish=finish,segments=segments)
        for j,finish in list(self.uploads.items()):
            if finish<=self.now+EPS:
                self.log('upload_complete',task_id=j)
                del self.uploads[j]

    @property
    def done(self):
        return super().done and not self.uploads

    def apply(self,p):
        s=self.snapshot()
        validate_plan(s,p)
        releases=report_releases(s,p.capture_times,p.routes)
        self.plan=p
        self.log('plan',routes=p.routes,edge_order=p.edge_order,
            edge_times={str(j):t for j,t in p.edge_times.items()},visible_ids=sorted(s.tasks),
            status_before=s.status,active_vehicles={str(i):None if v.active is None else asdict(v.active)
                for i,v in enumerate(s.vehicles)},edge_active=s.edge_active,planned_value=p.value,
            rule=p.rule,uplink=asdict(self.uplink),link_tails=s.link_tails)
        for vid,v in enumerate(self.vehicles):
            v.queue=list(p.routes[vid])
            if v.active or not v.queue:
                continue
            j=v.queue.pop(0);t=self.known[j]
            assert self.status[j]=='pending'
            leg=model.legacy.travel_time(v.x,v.y,t.x,t.y,v.speed)
            finish=self.now+leg+t.service
            v.active=model.legacy.ActiveJob(j,self.now,finish,leg)
            self.status[j]='active_capture'
            self.records[j].update(vehicle_id=vid,capture_start=self.now,travel_time=leg)
            self.log('capture_start',task_id=j,vehicle_id=vid,finish=finish,travel=leg)
        if self.edge_active is None and p.edge_order:
            j=p.edge_order[0]
            if self.status[j]=='ready':
                end=self.now+self.known[j].work*self.multiplier
                assert end<=self.known[j].deadline+EPS
                assert abs(p.edge_times[j][0]-self.now)<EPS
                self.edge_active=(j,self.now,end)
                self.status[j]='active_edge'
                self.records[j]['edge_start']=self.now
                self.log('edge_start',task_id=j,finish=end)
            else:
                self.log('edge_wait',task_id=j,ready_expected=releases[j])

    def advance(self):
        future=[]
        if self.arrival_index<len(self._stream):future.append(self._stream[self.arrival_index].release)
        future.extend(v.active.finish for v in self.vehicles if v.active)
        if self.edge_active:future.append(self.edge_active[2])
        future.extend(self.uploads.values())
        future.extend(self.known[j].deadline for j,st in self.status.items() if st not in ('completed','expired'))
        future=[t for t in future if t>self.now+EPS]
        if not future:
            if not self.done:raise RuntimeError('unfinished network episode without future event')
            return
        self.now=min(future);self._settle()

    def results(self):
        summary,rows,logs=super().results()
        committed=[r for r in rows if r.get('upload_finish') is not None]
        summary.update(upload_airtime=sum(sum(b-a for a,b in r['upload_segments']) for r in committed),
            upload_delay=sum(r['upload_finish']-r['capture_finish'] for r in committed),
            uploaded_packets=len(committed),uplink=asdict(self.uplink))
        return summary,rows,logs
