"""Exclusive GPU-window coverage and bounded host-submission gap attribution.

These are diagnostic intervals, not a counterfactual speedup or full-run timing.
In particular, copy-only activity is not automatically a removable dependency.
"""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

from analyze_stack_profile import duration, overlap, union


def analyze(trace, phase_metrics):
    events = trace['traceEvents']
    gpu = sorted((e for e in events if e.get('ph') == 'X' and
                  e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')), key=lambda e: e['ts'])
    assert gpu
    spans = lambda rows: [(e['ts'], e['ts']+e['dur']) for e in rows]
    compute = spans(e for e in gpu if e['cat'] == 'kernel')
    copies = spans(e for e in gpu if e['cat'] == 'gpu_memcpy')
    occupied = union(spans(gpu))
    start, end = occupied[0][0], occupied[-1][1]
    all_active = duration(occupied)
    compute_us = duration(compute)
    copy_only_us = duration(copies)-overlap(copies, compute)
    other_us = all_active-compute_us-copy_only_us
    idle_us = end-start-all_active
    runtime = {e['args']['correlation']: e for e in events if e.get('ph') == 'X'
               and e.get('cat') in ('cuda_runtime', 'cuda_driver')
               and 'correlation' in e.get('args', {})}
    starts = [e['ts'] for e in gpu]
    gaps = defaultdict(float)
    for (_, a), (b, _) in zip(occupied, occupied[1:]):
        following = gpu[bisect_left(starts, b)]
        api = runtime.get(following.get('args', {}).get('correlation'))
        if api is None:
            gaps['no_correlated_submission'] += b-a
            continue
        api_start, api_end = api['ts'], api['ts']+api['dur']
        gaps['before_next_api_call'] += max(0, min(b, api_start)-a)
        gaps['during_next_api_call'] += max(0, min(b, api_end)-max(a, api_start))
        gaps['after_next_api_return'] += max(0, b-max(a, api_end))
    assert abs(sum(gaps.values())-idle_us) < 1, (gaps, idle_us)
    sync = [e for e in events if e.get('ph') == 'X' and 'sync' in e.get('cat', '').lower()]
    sync_groups = defaultdict(lambda: {'count': 0, 'sum_ms': 0.0})
    for e in sync:
        item = sync_groups[e['name']]
        item['count'] += 1
        item['sum_ms'] += e['dur']/1000
    sync_spans = [(max(start, e['ts']), min(end, e['ts']+e['dur'])) for e in sync
                  if e['ts'] < end and e['ts']+e['dur'] > start]
    steps = sorted((e for e in events if e.get('ph') == 'X' and e.get('cat') == 'user_annotation'
                    and e.get('name', '').startswith('ProfilerStep#')), key=lambda e:e['ts'])
    numbers = [int(e['name'].partition('#')[2]) for e in steps]
    reference = [phase_metrics['step_s'][i] for i in numbers]
    classes = defaultdict(lambda: {'count': 0, 'sum_ms': 0.0})
    for e in gpu:
        if e['cat'] != 'kernel':
            continue
        name = e['name'].lower()
        group = ('attention' if 'fmha' in name else 'gemm' if 'gemm' in name or 'gemv' in name
                 else 'weight_dequantize' if 'dequantize' in name else 'casts_and_copies' if 'copy_kernel' in name
                 else 'other_compute')
        item = classes[group]; item['count'] += 1; item['sum_ms'] += e['dur']/1000
    coverage = {'compute_active': compute_us/1000, 'copies_without_compute': copy_only_us/1000,
                'other_gpu_activity': other_us/1000, 'no_gpu_activity': idle_us/1000}
    assert abs(sum(coverage.values())-(end-start)/1000) < .001
    return {'classification': 'DIAGNOSTIC ONLY', 'categories': dict(Counter(e.get('cat') for e in events)),
            'profiler_steps': numbers, 'gpu_window_ms': (end-start)/1000,
            'exclusive_gpu_window_ms': coverage,
            'no_gpu_activity_by_submission_ms': {k:v/1000 for k,v in gaps.items()},
            'cuda_sync_events_present': bool(sync), 'cuda_sync_by_name': dict(sync_groups),
            'cuda_sync_with_no_gpu_activity_ms': (duration(sync_spans)-overlap(sync_spans, occupied))/1000,
            'kernel_class_sums_not_additive_ms': dict(classes),
            'reference_phase_step_s': reference, 'reference_phase_sample_s': sum(reference),
            'profiled_cpu_step_s': [e['dur']/1e6 for e in steps],
            'profiled_cpu_sample_s': sum(e['dur']/1e6 for e in steps),
            'limitations': [
                'The domain is first-to-last recorded GPU activity, excluding CPU work outside those boundaries.',
                'Compute includes time concurrent with DMA. Copy-only means physical DMA activity without a recorded compute kernel; it does not prove all that time is removable.',
                'Before-next-API time is a host submission delay for the first following activity, including required CPU work and profiler overhead. It is not automatically removable Python overhead.',
                'After-API time can include dependencies, device scheduling or driver effects; absence of sync metadata cannot turn it into a transfer-wait measurement.',
                'Sync intervals can overlap computation, copies and host gaps; their sums are not another additive budget.',
                'Do not multiply a few variable-length steps into an asserted full-run time. Full-run estimates need explicit sampling/overhead assumptions.'
            ]}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); a = p.parse_args()
    with zipfile.ZipFile(a.raw) as z:
        manifest = json.loads(z.read('MANIFEST.json'))
        def read(name):
            data = z.read(name)
            assert hashlib.sha256(data).hexdigest() == manifest[name], name
            return data
        provenance = json.loads(read('PROFILE_PROVENANCE.json'))
        assert provenance['source_unchanged']
        for n, h in provenance['source_sha256'].items():
            assert hashlib.sha256(read(n)).hexdigest() == h
        data = read('timeline/timeline.json.gz')
        trace = json.loads(gzip.decompress(data) if data[:2] == b'\x1f\x8b' else data)
        phases = json.loads(read('phases/PARENT.json'))
        result = analyze(trace, phases['metrics'])
        result.update(commit=provenance['commit'], raw_sha256=hashlib.sha256(a.raw.read_bytes()).hexdigest())
        a.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        print(json.dumps({k:result[k] for k in ('exclusive_gpu_window_ms', 'no_gpu_activity_by_submission_ms',
                         'cuda_sync_events_present', 'reference_phase_sample_s', 'profiled_cpu_sample_s')}, indent=2))


if __name__ == '__main__':
    main()
