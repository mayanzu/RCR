"""Check the minimal current distribution; no historical raw logs required."""
from pathlib import Path
import hashlib
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from rosr import verify_frozen
from run_rosr_bench import source_hash


def main():
    manifest = json.loads((ROOT / 'latest_source_manifest.json').read_text(encoding='utf-8'))
    # The manifest describes the frozen core, not separately maintained
    # analysis, experiment, paper-build, or historical-source directories.
    core_sources = list(ROOT.glob('*.py'))
    for package in ('rosr', 'runtime'):
        core_sources.extend((ROOT / package).rglob('*.py'))
    actual = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(core_sources) if '__pycache__' not in p.parts}
    assert actual == manifest['files'], 'Retained Python source manifest mismatch'
    frozen = json.loads((ROOT / 'frozen_latest_policy.json').read_text(encoding='utf-8'))
    assert source_hash() == frozen['source_sha256'] == manifest['policy_sha256']
    for name, expected in frozen['entrypoint_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    assert hashlib.sha256((ROOT/'MODEL_OPTIMIZATION_PROTOCOL.md').read_bytes()).hexdigest() == frozen['protocol_sha256']
    contracts = verify_frozen()
    tests = unittest.defaultTestLoader.discover(str(ROOT), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=1).run(tests)
    assert result.wasSuccessful(), 'Focused tests failed'
    print(json.dumps(dict(source_files_verified=len(actual),
                          frozen_model_hashes_verified=len(contracts),
                          targeted_tests_passed=result.testsRun,
                          policy_sha256=source_hash(),
                          scope='Minimal retained source and mechanism tests; raw episodes are not distributed or re-audited by this package check.'), indent=2))


if __name__ == '__main__':
    main()
