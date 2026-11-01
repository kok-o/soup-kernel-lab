"""Screen a narrow transfer experiment, then run two untouched ABBA suites."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from run_offload_comparison import command, digest, raw_archive, snapshot, write_json


def save_evidence(out, roots, records, report):
    for name, root in roots.items():
        record = records[name]
        record['source_unchanged'] = all(digest((root/n).read_bytes()) == h
                                        for n, h in record['source_sha256'].items())
        assert record['source_unchanged'], 'Runtime snapshot mutated'
        write_json(root/'QUALIFICATION_PROVENANCE.json', record)
        report['results'].setdefault(name, {'commit': record['commit'], 'all_checks': False})
        report['results'][name]['raw_sha256'] = raw_archive(root, record, out/(name+'-raw-evidence.zip'))
    write_json(out/'COMPARISON.json', report)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--control', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--experiment', choices=('bulk', 'head4096'), default='bulk')
    a = p.parse_args()
    assert all(len(r) == 40 and all(c in '0123456789abcdef' for c in r)
               for r in (a.control, a.candidate))
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    roots = {n: out/n for n in ('control', 'candidate')}
    records = {n: snapshot(roots[n], ref) for n, ref in
               [('control', a.control), ('candidate', a.candidate)]}
    before, after = [records[n]['source_sha256'] for n in roots]
    changed = sorted(n for n in before.keys() | after.keys() if before.get(n) != after.get(n))
    allowed = {'bulk': ['solution/activation_offload.py', 'solution/test_activation_bulk_transfer.py'],
               'head4096': ['solution/__init__.py', 'solution/runtime_compile.py']}
    assert changed == allowed[a.experiment], changed
    assert all(b'patch_selective_offload(layers=28)' in (r/'solution/__init__.py').read_bytes() for r in roots.values())
    write_json(out/'COMPARISON_INPUTS.json', records)
    if a.prepare_only:
        print('Both pinned checkouts, all 15 official files and the narrow diff verified. No GPU used.')
        return
    names = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True)
    assert 'T4' in names, names
    write_json(out/'GPU_INVENTORY.json', {'names': names})
    report = {'experiment': a.experiment,
              'order': ['control screen', 'candidate screen', 'control ABBA', 'candidate ABBA'],
              'screen': {}, 'results': {},
              'limitations': 'Single-run screens are not qualification. Final results are sequential original 2+2 ABBA suites on one Kaggle T4 session; report base drift. All times include cold compilation. Profile timings are separate diagnostics.'}
    try:
        for name, root in roots.items():
            record = records[name]
            record['prepare_data_exit_code'] = command(root, [sys.executable, 'task/prepare_data.py'], 'qualification-data.log')
            assert record['prepare_data_exit_code'] == 0
            record['data_sha256'] = {p.relative_to(root).as_posix(): digest(p.read_bytes()) for p in (root/'data').glob('*.jsonl')}
            assert len(record['data_sha256']) >= 2
            if name == 'candidate':
                assert record['data_sha256'] == records['control']['data_sha256']
            env = os.environ.copy()
            env['SOUP_APPLY_SOLUTION'] = '1'
            record['pytest_exit_code'] = command(root, [sys.executable, '-m', 'pytest', 'solution', 'tests', '-q', '--tb=short'], 'qualification-tests.log', env)
            assert record['pytest_exit_code'] == 0, name+' tests failed'
            record['screen_exit_code'] = command(root, [sys.executable, 'harness/run_task.py', '--only', 'solution', '--repeats', '1', '--out', 'runs/screen'], 'qualification-screen.log')
            assert record['screen_exit_code'] == 0
            screen = json.loads((root/'runs/screen/result.json').read_text())
            report['screen'][name] = screen['summary']['solution']
            write_json(out/'SCREEN.json', report['screen'])
        b, c = [report['screen'][n] for n in roots]
        report['screen_pass'] = c['ok'] and c['gpu_peak_mib'] <= 500 and c['wall_s'] <= b['wall_s']+1
        if not report['screen_pass']:
            report['rejection'] = 'Screen missed <=500 MiB or was >1 s slower than fresh control. No full qualification started.'
            print(report['rejection'], flush=True)
            return
        for name, root in roots.items():
            record = records[name]
            record['run_task_exit_code'] = command(root, [sys.executable, 'harness/run_task.py', '--repeats', '2'], 'qualification-run.log')
            assert record['run_task_exit_code'] == 0, name+' original ABBA failed'
            record['check_rules_exit_code'] = command(root, [sys.executable, 'harness/check_rules.py'], 'qualification-checks.log')
            record['score_exit_code'] = command(root, [sys.executable, 'harness/score.py', 'runs/result.json'], 'qualification-score.log')
            checks = json.loads((root/'runs/checks.json').read_text())
            result = json.loads((root/'runs/result.json').read_text())
            qualified = (record['check_rules_exit_code'] == record['score_exit_code'] == 0
                         and len(checks['checks']) == 10 and all(x['ok'] is True for x in checks['checks']))
            report['results'][name] = {'commit': record['commit'], 'all_checks': qualified,
                                      'summary': result['summary'], 'runs': result['runs']}
            save_evidence(out, roots, records, report)
            assert qualified, name+' official checks failed'
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        save_evidence(out, roots, records, report)
        print(json.dumps({k: report[k] for k in ('screen', 'results')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
