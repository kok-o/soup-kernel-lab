"""Verify downloaded original-harness evidence without importing GPU libraries."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify(raw, repo, expected_commit):
    with zipfile.ZipFile(raw) as z:
        manifest = json.loads(z.read('MANIFEST.json'))
        assert set(z.namelist()) == set(manifest) | {'MANIFEST.json'}
        assert all(sha(z.read(n)) == h for n, h in manifest.items())
        provenance = json.loads(z.read('QUALIFICATION_PROVENANCE.json'))
        assert provenance['commit'] == expected_commit
        assert provenance['source_unchanged'] is True
        for name in ('pytest_exit_code', 'run_task_exit_code', 'check_rules_exit_code', 'score_exit_code'):
            assert provenance[name] == 0, (name, provenance.get(name))
        expected = provenance['source_sha256']
        for name, h in expected.items():
            data = z.read(name)
            assert sha(data) == h
            assert data == subprocess.check_output(['git', '-C', str(repo), 'show', expected_commit + ':' + name])
        baseline = json.loads(subprocess.check_output(['git', '-C', str(repo), 'show', expected_commit + ':BASELINE_PROVENANCE.json']))
        official = {n: h for n, h in baseline['source_sha256'].items()
                    if n.startswith(('harness/', 'task/', 'tests/'))}
        assert len(official) == 15
        assert all(expected.get(n) == h for n, h in official.items())
        checks = json.loads(z.read('runs/checks.json'))
        result = json.loads(z.read('runs/result.json'))
        assert checks['result_sha256'] == sha(z.read('runs/result.json'))
        solution_hash = hashlib.sha256()
        for name in sorted(n for n in expected if n.startswith('solution/')):
            solution_hash.update(name[len('solution/'):].encode())
            solution_hash.update(z.read(name))
        assert checks['solution_sha256'] == solution_hash.hexdigest()
        assert len(checks['checks']) == 10 and all(c['ok'] is True for c in checks['checks'])
        rows = []
        for side in ('base', 'solution'):
            runs = result['runs'][side]
            assert len(runs) == 2
            for r in runs:
                m = json.loads(z.read('runs/' + r['run'] + '/metrics.json'))
                assert r['exit_code'] == m['exit_code'] == 0 and m['steps'] == 64
                assert len(m['step_s']) == 64
                assert m['cuda_device'] == 'Tesla T4' and m['device_count'] == 1
                assert m['visible_devices'] == '0'
                assert m['triton_watch_error'] is None
                assert m['alloc_conf'] == m['alloc_conf_expected'] == 'expandable_segments:True'
                assert z.read('runs/' + r['run'] + '/adapter/rewind.jsonl').strip()
                row = {k: r[k] for k in ('run', 'wall_s', 'gpu_peak_mib')}
                row.update({k: m[k] for k in ('train_loop_s', 'apply_s', 'torch_max_allocated_mib', 'torch_max_reserved_mib')})
                rows.append(row)
            summary = result['summary'][side]
            assert abs(summary['wall_s'] - statistics.median(r['wall_s'] for r in runs)) <= 0.011
            assert summary['gpu_peak_mib'] == max(r['gpu_peak_mib'] for r in runs)
        return {'commit': expected_commit, 'raw_sha256': sha(raw.read_bytes()), 'all_checks': 10,
                'source_files': len(expected), 'official_files': len(official),
                'summary': result['summary'], 'versions': result['versions'], 'runs': rows,
                'data_sha256': provenance['data_sha256'],
                'own_kernel_share': [c['detail'] for c in checks['checks'] if c['check'].startswith('4.')],
                'tests': z.read('qualification-tests.log').decode()[-1800:]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--downloads', type=Path, required=True)
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    filenames = list(a.downloads.rglob('COMPARISON.json'))
    assert len(filenames) == 1, filenames
    report = json.loads(filenames[0].read_text())
    verified = {}
    for name in ('control', 'candidate'):
        files = list(a.downloads.rglob(name + '-raw-evidence.zip'))
        assert len(files) == 1, files
        expected = report['results'][name]
        assert expected['all_checks'] is True
        assert sha(files[0].read_bytes()) == expected['raw_sha256']
        verified[name] = verify(files[0], a.repo, expected['commit'])
        assert verified[name]['summary'] == expected['summary']
    assert verified['control']['data_sha256'] == verified['candidate']['data_sha256']
    base_times = [r['wall_s'] for v in verified.values() for r in v['runs'] if r['run'].startswith('base')]
    verified['limitations'] = report['limitations']
    verified['all_base_times_s'] = base_times
    verified['base_range_s'] = max(base_times) - min(base_times)
    a.out.write_text(json.dumps(verified, indent=2) + '\n')
    print(json.dumps({n: v['summary'] for n, v in verified.items() if isinstance(v, dict)}, indent=2))


if __name__ == '__main__':
    main()
