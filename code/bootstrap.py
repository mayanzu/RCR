"""Package paths, source identity and external run outputs.

Core scheduling and simulation modules do not import command-line runners.
"""
from pathlib import Path
import hashlib
import sys

CODE_ROOT=Path(__file__).resolve().parent
PACKAGE_ROOT=CODE_ROOT.parent
BASELINES_ROOT=CODE_ROOT/'baselines'
SCRIPTS_ROOT=PACKAGE_ROOT/'scripts'
TESTS_ROOT=PACKAGE_ROOT/'tests'
RUNS_ROOT=PACKAGE_ROOT.parent/(PACKAGE_ROOT.name+'_runs')
for directory in (SCRIPTS_ROOT,CODE_ROOT):
    if str(directory) not in sys.path:sys.path.insert(0,str(directory))
sys.dont_write_bytecode=True


def source_hash():
    digest=hashlib.sha256()
    for source in sorted((CODE_ROOT/'rosr').glob('*.py')):
        digest.update(source.name.encode());digest.update(source.read_bytes())
    return digest.hexdigest()


def output_directory(value=None,label='run'):
    path=(Path(value) if value is not None else RUNS_ROOT/label).resolve()
    if path.is_relative_to(PACKAGE_ROOT.resolve()):
        raise ValueError('Run output must be outside the code-only package')
    return path
