"""Current simulator bridge and dependency integrity checks."""
from pathlib import Path
import hashlib
import model
import audit
HERE = Path(__file__).resolve().parent
EXPECTED = {'model.py': '4568e58e0f9f3e2db63550ac01c06b562a2b3a951750e3d457a1da1d9aec32ff', 'routing.py': '23164053f0ce13200b3d8cdf657786e63e9a19f35b94955c0e62ab36a03c7a17', 'audit.py': '492b1ccf02ac179db663c761fa9bfa9dc7b1ca45b8aee9bb262eaad3ea149285'}

def verify_frozen():
    for name, expected in EXPECTED.items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected, name
    return {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in EXPECTED}

def action_key(s, p):
    """Only irreversible starts. Waiting on distinct future heads is identical.

    The executor has no plan-dependent alarm. Density continuation rebuilds all tails.
    """
    starts = tuple((i, route[0]) for i, route in enumerate(p.routes)
                   if route and s.vehicles[i].active is None)
    edge = p.edge_order[0] if (s.edge_active is None and p.edge_order
                             and s.status[p.edge_order[0]] == 'ready') else None
    return starts, edge
