"""Verify released Protected-NS observations, fingerprints and paired statistics."""
from pathlib import Path
import csv
import hashlib
import json
import math
from collections import defaultdict
from run_protected_nosearch import paired_stats

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

def main():
    metadata = json.loads((HERE/'protected_nosearch_metadata.json').read_text(encoding='utf-8'))
    for name, digest in metadata['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    records = list(csv.DictReader((HERE/'protected_nosearch_observations.csv').open(encoding='utf-8',newline='')))
    assert len(records) == metadata['observations'] == 720
    groups = defaultdict(dict)
    for row in records:
        key = (row['condition'], row['method'])
        seed = int(row['seed'])
        assert seed not in groups[key], (key, seed)
        value = float(row['value_rate_percent'])
        assert math.isfinite(value) and 0 <= value <= 100
        groups[key][seed] = value
    summary = json.loads((HERE/'protected_nosearch_summary.json').read_text(encoding='utf-8'))
    comparisons = 0
    for family, conditions in summary.items():
        for name, entry in conditions.items():
            condition = 'hefei_'+name if family == 'hefei' else name
            digits = 10 if family == 'hefei' else 4
            for label, first, second in [('protected_ns_minus_density','protected_ns','density'),
                                         ('rcr_minus_protected_ns','rosr','protected_ns')]:
                if label not in entry:
                    continue
                a,b = groups[condition,first], groups[condition,second]
                assert len(a) == len(b) == 30 and set(a) == set(b)
                seeds = sorted(a)
                actual = paired_stats([a[s] for s in seeds],[b[s] for s in seeds],percent_decimals=digits)
                for key, value in actual.items():
                    expected = entry[label][key]
                    assert (value == expected if isinstance(value,str) else math.isclose(value,expected,rel_tol=1e-12,abs_tol=1e-12)), (condition,label,key,value,expected)
                comparisons += 1
    print(json.dumps(dict(status='PASS',observations=len(records),paired_comparisons=comparisons,
                         source_fingerprints=len(metadata['source_sha256'])),indent=2))

if __name__ == '__main__':
    main()
