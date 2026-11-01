"""Run two immutable source snapshots with the original Kaggle T4 harness.

The driver never imports torch in its own process, patches the meter, or changes
the official files. --prepare-only permits a CPU checkout/provenance check.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

REPOSITORY = 'https://github.com/kok-o/soup-kernel-lab.git'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def snapshot(root, ref):
    if root.exists():
        raise RuntimeError(f'Use a fresh output directory: {root}')
    subprocess.run(['git', 'init', str(root)], check=True, stdout=subprocess.DEVNULL)
    for key, value in [('core.autocrlf', 'false'), ('core.eol', 'lf')]:
        subprocess.run(['git', '-C', str(root), 'config', key, value], check=True)
    subprocess.run(['git', '-C', str(root), 'remote', 'add', 'origin', REPOSITORY], check=True)
    subprocess.run(['git', '-C', str(root), 'fetch', '--depth', '1', 'origin', ref], check=True)
    subprocess.run(['git', '-C', str(root), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
    names = git(root, 'ls-files', '-z', '--', 'solution', 'harness', 'task', 'tests').split('\0')
    expected = {n: digest((root / n).read_bytes()) for n in names if n}
    baseline = json.loads((root / 'BASELINE_PROVENANCE.json').read_text())
    official = {n: h for n, h in baseline['source_sha256'].items()
                if n.startswith(('harness/', 'task/', 'tests/'))}
    assert len(official) == 15
    assert all(expected.get(n) == h for n, h in official.items()), 'Official source changed'
    return {'repository': REPOSITORY, 'requested_ref': ref, 'commit': git(root, 'rev-parse', 'HEAD'),
            'source_sha256': expected, 'protected_sha256': official}


def command(root, args, name, env=None):
    print(f'[{root.name}] {name}: start', flush=True)
    started = time.perf_counter()
    with (root / name).open('w', encoding='utf-8') as log:
        proc = subprocess.Popen(args, cwd=root, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors='replace')
        for line in proc.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
        rc = proc.wait()
    print(f'[{root.name}] {name}: exit={rc}, orchestration={time.perf_counter()-started:.2f}s', flush=True)
    return rc


def raw_archive(root, record, out):
    files = {n: root / n for n in record['source_sha256']}
    runs = root / 'runs'
    if runs.exists():
        for p in runs.rglob('*'):
            if not p.is_file() or any(part in ('triton_cache', 'inductor_cache', '__pycache__')
                                     for part in p.relative_to(root).parts):
                continue
            if p.suffix in ('.json', '.jsonl', '.yaml', '.log') and not p.name.startswith('tokenizer'):
                files[p.relative_to(root).as_posix()] = p
    for p in root.glob('qualification-*'):
        if p.is_file():
            files[p.name] = p
    files['QUALIFICATION_PROVENANCE.json'] = root / 'QUALIFICATION_PROVENANCE.json'
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        manifest = {}
        for name, p in sorted(files.items()):
            data = p.read_bytes()
            z.writestr(name, data)
            manifest[name] = digest(data)
        z.writestr('MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    return digest(out.read_bytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--control', required=True)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--prepare-only', action='store_true')
    args = ap.parse_args()
    for ref in (args.control, args.candidate):
        assert len(ref) == 40 and all(c in '0123456789abcdef' for c in ref), 'Pin full commits'
    assert args.control != args.candidate
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    roots = {n: out / n for n in ('control', 'candidate')}
    records = {n: snapshot(roots[n], ref) for n, ref in
               [('control', args.control), ('candidate', args.candidate)]}
    a, b = (records[n]['source_sha256'] for n in ('control', 'candidate'))
    changed = [n for n in a.keys() | b.keys() if a.get(n) != b.get(n)]
    assert changed == ['solution/__init__.py'], changed
    before = (roots['control'] / changed[0]).read_bytes()
    after = (roots['candidate'] / changed[0]).read_bytes()
    assert before.count(b'patch_selective_offload(layers=20)') == 1
    assert before.replace(b'patch_selective_offload(layers=20)', b'patch_selective_offload(layers=28)') == after
    write_json(out / 'COMPARISON_INPUTS.json', records)
    print('Exactly one runtime change: activation offload 20 -> 28 layers.', flush=True)
    if args.prepare_only:
        print('CPU checkout/provenance verification complete; no training requested.', flush=True)
        return
    gpu_names = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True)
    assert 'T4' in gpu_names, gpu_names
    write_json(out / 'GPU_INVENTORY.json', {'names': gpu_names})
    comparison = {'experiment': 'offload20 vs offload28, slot-view cache fixed',
                  'order': ['control', 'candidate'], 'results': {},
                  'limitations': 'Sequential original 2+2 ABBA suites on one Kaggle session; report base drift. Historical 61.40s is not the new control time.'}
    # Every torch-using command is a child process and ends before measurements.
    for name in ('control', 'candidate'):
        root, record = roots[name], records[name]
        try:
            record['prepare_data_exit_code'] = command(root, [sys.executable, 'task/prepare_data.py'], 'qualification-data.log')
            assert record['prepare_data_exit_code'] == 0, 'Data preparation failed'
            record['data_sha256'] = {p.relative_to(root).as_posix(): digest(p.read_bytes())
                                     for p in (root / 'data').glob('*.jsonl')}
            assert len(record['data_sha256']) >= 2
            if name == 'candidate':
                assert record['data_sha256'] == records['control']['data_sha256'], 'Data differ'
            env = os.environ.copy()
            env['SOUP_APPLY_SOLUTION'] = '1'
            record['pytest_exit_code'] = command(root, [sys.executable, '-m', 'pytest', 'solution', 'tests', '-q', '--tb=short'], 'qualification-tests.log', env=env)
            assert record['pytest_exit_code'] == 0, 'Patched tests failed'
            record['run_task_exit_code'] = command(root, [sys.executable, 'harness/run_task.py', '--repeats', '2'], 'qualification-run.log')
            assert record['run_task_exit_code'] == 0, 'Original ABBA failed'
            record['check_rules_exit_code'] = command(root, [sys.executable, 'harness/check_rules.py'], 'qualification-checks.log')
            record['score_exit_code'] = command(root, [sys.executable, 'harness/score.py', 'runs/result.json'], 'qualification-score.log')
            checks = json.loads((root / 'runs/checks.json').read_text())
            result = json.loads((root / 'runs/result.json').read_text())
            qualified = (record['check_rules_exit_code'] == record['score_exit_code'] == 0 and
                         len(checks['checks']) == 10 and all(c['ok'] is True for c in checks['checks']))
            comparison['results'][name] = {'commit': record['commit'], 'all_checks': qualified,
                                           'summary': result['summary'], 'runs': result['runs']}
            assert qualified, 'Official checks failed'
        except Exception as exc:
            record['error'] = repr(exc)
            comparison['results'].setdefault(name, {'commit': record['commit'], 'all_checks': False, 'error': repr(exc)})
        finally:
            observed = {n: digest((root / n).read_bytes()) for n in record['source_sha256']}
            record['source_unchanged'] = observed == record['source_sha256']
            write_json(root / 'QUALIFICATION_PROVENANCE.json', record)
            archive = out / f'{name}-raw-evidence.zip'
            comparison['results'][name]['raw_sha256'] = raw_archive(root, record, archive)
            write_json(out / 'COMPARISON.json', comparison)
        assert record['source_unchanged'], 'Source mutated during run'
    print(json.dumps({n: {'all_checks': r['all_checks'], 'summary': r.get('summary')}
                      for n, r in comparison['results'].items()}, indent=2), flush=True)
    assert all(r['all_checks'] for r in comparison['results'].values()), 'One or more variants failed; evidence saved'


if __name__ == '__main__':
    main()
