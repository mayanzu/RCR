"""Verify the code-only source tree and run its mechanism tests.

    python -m pip install -r requirements.txt
    python -B scripts/verify_package.py
    python -B scripts/run_rosr.py --seed 20460400 --profile tighter_deadlines
"""
from pathlib import Path
import hashlib
import inspect
import json
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from bootstrap import PACKAGE_ROOT,TESTS_ROOT,source_hash
from frozen_metadata import EXPECTED_SOURCES,FROZEN_POLICY,PRESERVED_SOURCE_HASHES
from rosr import RosrPolicy,ALGORITHM_VERSION,verify_frozen


def main():
    actual={p.relative_to(PACKAGE_ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in ('code','scripts','tests')
            for p in sorted((PACKAGE_ROOT/directory).rglob('*.py'))
            if '__pycache__' not in p.parts and p.name!='frozen_metadata.py'}
    assert actual==EXPECTED_SOURCES,'Retained Python source mismatch'
    for name,digest in PRESERVED_SOURCE_HASHES.items():
        assert actual[name]==digest,'Original algorithm/model/control changed: '+name
    assert source_hash()==FROZEN_POLICY['source_sha256'],'Frozen algorithm changed'
    assert ALGORITHM_VERSION==FROZEN_POLICY['version']
    parameters=inspect.signature(RosrPolicy.__init__).parameters
    for name in ('beta','max_depth','seconds','compact_ties','conflict_cuts'):
        assert parameters[name].default==FROZEN_POLICY[name],name
    contracts=verify_frozen()
    tests=unittest.defaultTestLoader.discover(str(TESTS_ROOT),pattern='test_*.py')
    result=unittest.TextTestRunner(verbosity=1).run(tests)
    assert result.wasSuccessful(),'Mechanism tests failed'
    print(json.dumps(dict(source_files_verified=len(actual),
        preserved_algorithm_model_control_files=len(PRESERVED_SOURCE_HASHES),
        frozen_model_hashes_verified=len(contracts),targeted_tests_passed=result.testsRun,
        policy_sha256=source_hash(),scope='Code-only architecture; original core/model/control bytes preserved.'),indent=2))


if __name__=='__main__':main()
