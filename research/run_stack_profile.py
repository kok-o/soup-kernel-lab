"""Acquire diagnostic traces; never qualify the observer's timings or peaks."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

from run_offload_comparison import snapshot


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--commit', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--modes', nargs='+', choices=('phases', 'timeline', 'memory'), default=['phases', 'timeline', 'memory'])
    a = p.parse_args()
    assert len(a.commit) == 40 and all(c in '0123456789abcdef' for c in a.commit)
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = out / 'source'
    record = snapshot(root, a.commit)
    expected = record['source_sha256']
    assert b'patch_selective_offload(layers=28)' in (root / 'solution/__init__.py').read_bytes()
    observer = Path(__file__).with_name('profile_stack.py')
    record.update(classification='DIAGNOSTIC ONLY, not qualification', observer_sha256=sha(observer.read_bytes()),
                  driver_sha256=sha(Path(__file__).read_bytes()), results={})
    (out / 'PROFILE_PROVENANCE.json').write_text(json.dumps(record, indent=2))
    if a.prepare_only:
        print('Profile source and all15officialfiles verified; no GPU requested.')
        return
    subprocess.run([sys.executable, str(root / 'task/prepare_data.py')], cwd=root, check=True)
    sys.path.insert(0, str(root / 'harness'))
    from run_task import warm_up
    warm_up(out)
    for mode in a.modes:
        directory = out / mode
        print('DIAGNOSTIC MODE:', mode, flush=True)
        rc = subprocess.run([sys.executable, str(observer), '--source', str(root), '--out', str(directory), '--mode', mode], cwd=root).returncode
        record['results'][mode] = {'exit_code': rc}
        if (directory / 'PARENT.json').exists():
            record['results'][mode]['parent'] = json.loads((directory / 'PARENT.json').read_text())
        observed = {n: sha((root / n).read_bytes()) for n in expected}
        assert observed == expected, 'Protected runtime snapshot mutated'
        record['source_unchanged'] = True
        (out / 'PROFILE_PROVENANCE.json').write_text(json.dumps(record, indent=2))
    files = {n: root / n for n in expected}
    files['research/profile_stack.py'] = observer
    files['research/run_stack_profile.py'] = Path(__file__)
    files['PROFILE_PROVENANCE.json'] = out / 'PROFILE_PROVENANCE.json'
    files['warmup.log'] = out / 'warmup.log'
    for mode in a.modes:
        for path in (out / mode).iterdir():
            if path.is_file() and (path.suffix in ('.json', '.log', '.txt', '.yaml') or path.name.endswith('.json.gz')):
                files[mode + '/' + path.name] = path
    archive = out / 'stack-profile-raw.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        manifest = {}
        for name, path in sorted(files.items()):
            data = path.read_bytes()
            z.writestr(name, data)
            manifest[name] = sha(data)
        z.writestr('MANIFEST.json', json.dumps(manifest, indent=2))
    print('RAW PROFILE:', archive, sha(archive.read_bytes()), flush=True)
    print(json.dumps({m: r['exit_code'] for m, r in record['results'].items()}), flush=True)
    assert all(r['exit_code'] == 0 for r in record['results'].values()), 'Diagnostic failed; raw logs retained'


if __name__ == '__main__':
    main()
