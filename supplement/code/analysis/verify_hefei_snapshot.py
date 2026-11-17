"""Replay the Hefei case with its saved, fingerprinted source snapshot.

The current package may gain harmless entry points or a refreshed source
manifest after the study. The original case verifier correctly rejects such
changes; this runner stages the retained source snapshot and the recorded
case/data/traces in an isolated temporary tree, then runs the original verifier.

Run from the project root: python -B code/analysis/verify_hefei_snapshot.py
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
CASE=ROOT/'opt/osm_hefei'
CASE_SOURCES=ROOT/'code/history/osm_hefei_original'
ARCHIVE=CASE/'results/confirmation/frozen_code_snapshot.zip'
OUTPUT=Path(__file__).resolve().parent/'hefei_snapshot_replay.json'


def main():
    with tempfile.TemporaryDirectory(prefix='vcs_hefei_snapshot_') as tmp:
        stage=Path(tmp);code=stage/'code';case=stage/'opt/osm_hefei'
        code.mkdir(parents=True);case.mkdir(parents=True)
        with zipfile.ZipFile(ARCHIVE) as archive:
            for info in archive.infolist():
                target=(code/info.filename).resolve()
                if not target.is_relative_to(code.resolve()) or info.is_dir():
                    raise ValueError(f'Unsafe or unexpected archive member: {info.filename}')
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(archive.read(info))
        for item in CASE_SOURCES.glob('*.py'): shutil.copy2(item,case/item.name)
        shutil.copy2(CASE/'protocol.json',case/'protocol.json')
        shutil.copytree(CASE/'data',case/'data')
        shutil.copytree(CASE/'results/confirmation',case/'results/confirmation')
        # The original verifier intentionally includes machine/OS identity in
        # its experiment fingerprint. For trace replay on another machine,
        # compare every non-hardware setting to the recorded configuration,
        # independently verify the recorded fingerprint, and disclose the
        # changed hardware instead of pretending the machine is identical.
        verifier=case/'verify_case.py'
        source=verifier.read_text(encoding='utf8')
        old='''    _, current_fingerprint = configuration(args.stage)
    if current_fingerprint != settings["fingerprint"]:
        raise AssertionError("Current source, data or environment differs from the experiment fingerprint")'''
        new='''    current, _ = configuration(args.stage)
    recorded = {key: settings[key] for key in current}
    for key in current:
        if key != "environment" and current[key] != recorded[key]:
            raise AssertionError(f"Source, protocol or data mismatch: {key}")
    for key in ("python", "numpy", "scipy", "threads"):
        if current["environment"][key] != recorded["environment"][key]:
            raise AssertionError(f"Runtime dependency or thread setting mismatch: {key}")
    recorded_fingerprint = hashlib.sha256(json.dumps(recorded, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    if recorded_fingerprint != settings["fingerprint"]:
        raise AssertionError("Recorded settings do not match the stored experiment fingerprint")'''
        if source.count(old)!=1:
            raise RuntimeError('Original verifier no longer matches the expected fingerprint guard.')
        verifier.write_text(source.replace(old,new,1),encoding='utf8',newline='\n')
        env=dict(os.environ)
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            env[key]='1'
        result=subprocess.run([sys.executable,'-B',str(case/'verify_case.py'),'--stage','confirmation'],
                              cwd=stage,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              encoding='utf8',errors='replace',text=True)
        if result.returncode:
            raise RuntimeError(f'Frozen-snapshot replay failed:\n{result.stdout[-5000:]}')
        report=json.loads((case/'results/confirmation/verification.json').read_text(encoding='utf8'))
        expected=json.loads((CASE/'results/confirmation/settings.json').read_text(encoding='utf8'))
        assert report['result']=='PASS' and report['episodes']==450
        assert report['evidence_fingerprint']==expected['fingerprint']
        report['replay_mode']='retained source snapshot, isolated staging, hardware-independent trace audit'
        report['recorded_hardware']={k:expected['environment'][k] for k in ('platform','processor')}
        report['replay_hardware']={'platform':platform.platform(),'processor':platform.processor()}
        report['scope']='Verifies source/data fingerprints and traces; does not reproduce historical wall times.'
        OUTPUT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
        print(json.dumps(report,indent=2))


if __name__=='__main__': main()
