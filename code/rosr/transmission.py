"""Public fixed-share uplink calendar and path-separable report releases.

This module changes no frozen source. A packet can pause at slot boundaries;
vehicle acquisition and packet transmission use different resources.
"""
from dataclasses import dataclass
import math
from .boot import model, EPS


@dataclass(frozen=True)
class Uplink:
    vehicles: int
    slot: float = .025
    rate: float = 1.

    def __post_init__(self):
        if self.vehicles < 1 or self.slot <= 0 or not 0 < self.rate <= 1:
            raise ValueError('positive fleet/slot and 0 < normalized rate <= 1 required')

    def transmit(self, vehicle, earliest, work):
        """Earliest FCFS service using only this vehicle's reserved slots."""
        if not 0 <= vehicle < self.vehicles or work < 0 or earliest < -EPS:
            raise ValueError('invalid transmission request')
        cursor, remaining, segments = max(0., earliest), work/self.rate, []
        period = self.slot*self.vehicles
        while remaining > 1e-12:
            cycle = max(0, math.floor(cursor/period))
            start = cycle*period + vehicle*self.slot
            end = start+self.slot
            if cursor >= end-1e-12:
                start += period
                end += period
            begin = max(cursor, start)
            duration = min(remaining, end-begin)
            if duration <= 1e-12:
                cursor = end+1e-11
                continue
            cursor = begin+duration
            segments.append((begin, cursor))
            remaining -= duration
        return cursor, tuple(segments)

    def finish_time(self, vehicle, earliest, work):
        """O(1) completion prediction; execution still enumerates actual slots."""
        if not 0 <= vehicle < self.vehicles or work < 0 or earliest < -EPS:
            raise ValueError('invalid transmission request')
        earliest=max(0.,earliest)
        service=work/self.rate
        if service<=1e-12:return earliest
        period=self.slot*self.vehicles
        cycle=max(0,math.floor(earliest/period))
        start=cycle*period+vehicle*self.slot
        end=start+self.slot
        if earliest>=end-1e-12:
            start+=period;end+=period
        begin=max(earliest,start)
        available=end-begin
        if service<=available+1e-12:return begin+service
        remaining=service-available
        windows=max(1,math.ceil((remaining-1e-12)/self.slot))
        return start+windows*period+remaining-(windows-1)*self.slot


def route_releases(s, vehicle, captures):
    """Predict this route after immutable upload and acquisition commitments."""
    uplink = getattr(s, 'uplink', None)
    if uplink is None:
        return {j:c+s.tasks[j].upload for j,c in captures.items()}
    tail = max(s.now, s.link_tails[vehicle])
    v = s.vehicles[vehicle]
    if v.active is not None and s.status[v.active.task_id] != 'expired':
        j = v.active.task_id
        tail = uplink.finish_time(vehicle, max(tail,v.active.finish), s.tasks[j].upload)
    reports = {}
    for j,c in captures.items():
        tail = uplink.finish_time(vehicle,max(tail,c),s.tasks[j].upload)
        reports[j] = tail
    return reports


def report_releases(s, captures, routes=None):
    if getattr(s,'uplink',None) is None:
        return model.report_releases(s,captures)
    reports = {j:max(s.now,s.ready_times[j]) for j,st in s.status.items()
               if st in ('ready','uploading')}
    for vid,v in enumerate(s.vehicles):
        if v.active is not None and s.status[v.active.task_id] != 'expired':
            j=v.active.task_id
            reports[j]=s.uplink.finish_time(vid,max(s.now,s.link_tails[vid],v.active.finish),s.tasks[j].upload)
    if captures and routes is None:
        raise ValueError('route ownership is required for a shared-uplink prediction')
    if routes is not None:
        for vid,route in enumerate(routes):
            reports.update(route_releases(s,vid,{j:captures[j] for j in route}))
    return reports
