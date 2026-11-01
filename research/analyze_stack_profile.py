"""Analyze diagnostic intervals and allocation history without summing overlaps."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import zipfile


def union(intervals):
    result = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if result and a <= result[-1][1]:
            result[-1][1] = max(b, result[-1][1])
        else:
            result.append([a, b])
    return result


def duration(intervals):
    return sum(b-a for a, b in union(intervals))


def overlap(a, b):
    a, b = union(a), union(b)
    i = j = 0
    total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1])-max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def phases(data, parent):
    rows = [r for r in data['spans'] if 'end_s' in r]
    begin = [r['end_s'] for r in rows if r['name'] == 'harness.step.begin']
    end = [r['end_s'] for r in rows if r['name'] == 'harness.step.end']
    assert len(begin) == len(end) == 64, (len(begin), len(end))
    first, last = min(begin), max(end)
    main = [r for r in rows if r['thread_name'] == 'MainThread']
    finish = max(r['end_s'] for r in main)
    points = {0.0, first, last, finish}
    for r in main:
        points.update((r['start_s'], r['end_s']))
    times = sorted(points)
    groups = defaultdict(float)
    def category(name):
        if name.startswith('import.'):
            return 'imports'
        if name.startswith('data.'):
            return 'data_and_fingerprint'
        if name.startswith('model.'):
            return 'model_load_quantize_pin'
        if name.startswith('save.'):
            return 'save'
        if name.startswith('compile.') or name == 'solution.apply':
            return 'apply_and_compile_preparation'
        if name == 'Trainer.init':
            return 'trainer_construction'
        return 'other_setup_or_completion'
    for a, b in zip(times, times[1:]):
        mid = (a+b)/2
        if first <= mid <= last:
            groups['training_interval'] += b-a
            continue
        containing = [r for r in main if r['start_s'] <= mid < r['end_s']]
        name = max(containing, key=lambda r:r['start_s'])['name'] if containing else ''
        groups[category(name)] += b-a
    aggregates = defaultdict(lambda: {'count': 0, 'inclusive_s': 0, 'intervals': []})
    for r in rows:
        a = aggregates[r['name']]
        a['count'] += 1
        a['inclusive_s'] += r['end_s']-r['start_s']
        a['intervals'].append((r['start_s'], r['end_s']))
    for a in aggregates.values():
        a['wall_union_s'] = duration(a.pop('intervals'))
    return {'main_thread_partition_s': dict(groups), 'partition_total_s': sum(groups.values()),
            'first_step_s': first, 'last_step_s': last, 'aggregates': dict(aggregates),
            'late_finalization_plus_poller_s': parent['child_atexit_to_parent_return_s'],
            'diagnostic_metrics': parent['metrics'], 'warnings': data['warnings'],
            'limits': 'Inclusive spans overlap and worker compilation can overlap main-thread work. Only the partition is additive; diagnostic overhead remains included.'}


def timeline(data):
    events = data['traceEvents']
    gpu = [e for e in events if e.get('ph') == 'X' and e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
    assert gpu, 'No CUDA activities captured'
    steps = [e for e in events if e.get('ph') == 'X' and e.get('name', '').startswith('ProfilerStep#')]
    start = min(e['ts'] for e in gpu)
    stop = max(e['ts']+e['dur'] for e in gpu)
    ranges = defaultdict(list)
    for e in events:
        if e.get('ph') == 'X' and e.get('cat') == 'user_annotation' and e.get('name', '').startswith(('NF4.', 'Head.', 'Activation.')):
            ranges[e['tid']].append(e)
    for r in ranges.values():
        r.sort(key=lambda e:e['ts'])
    starts = {tid: [e['ts'] for e in rows] for tid, rows in ranges.items()}
    runtime = {e.get('args', {}).get('correlation'): e for e in events
               if e.get('ph') == 'X' and e.get('cat') in ('cuda_runtime', 'cuda_driver') and 'correlation' in e.get('args', {})}
    def region(e):
        call = runtime.get(e.get('args', {}).get('correlation'))
        if call is None:
            return 'unattributed'
        rows = ranges.get(call['tid'], [])
        i = bisect_right(starts.get(call['tid'], []), call['ts'])-1
        for r in reversed(rows[max(0, i-60):i+1]):
            if r['ts'] <= call['ts'] <= r['ts']+r['dur']:
                return r['name']
        return 'unattributed'
    buckets = defaultdict(list)
    by_name = defaultdict(lambda: {'count':0, 'sum_us':0, 'bytes':0})
    by_region = defaultdict(lambda: {'count':0, 'sum_us':0, 'h2d_bytes':0, 'd2h_bytes':0})
    for e in gpu:
        span = (e['ts'], e['ts']+e['dur'])
        if e['cat'] == 'kernel':
            kind = 'compute'
        elif 'htod' in e['name'].lower():
            kind = 'h2d'
        elif 'dtoh' in e['name'].lower():
            kind = 'd2h'
        else:
            kind = 'other_gpu'
        buckets[kind].append(span)
        b = e.get('args', {}).get('bytes', 0)
        b = b if isinstance(b, (int, float)) else 0
        agg = by_name[e['name']]
        agg['count'] += 1; agg['sum_us'] += e['dur']; agg['bytes'] += b
        agg = by_region[region(e)]
        agg['count'] += 1; agg['sum_us'] += e['dur']
        if kind in ('h2d', 'd2h'):
            agg[kind+'_bytes'] += b
    occupied = union([(e['ts'], e['ts']+e['dur']) for e in gpu])
    gaps = []
    for (_, a), (b, _) in zip(occupied, occupied[1:]):
        if b-a >= 50:
            next_e = min((e for e in gpu if e['ts'] >= b-0.001), key=lambda e:e['ts'])
            launch = runtime.get(next_e.get('args', {}).get('correlation'))
            gaps.append({'start_us':a, 'duration_us':b-a, 'next_activity':next_e['name'],
                         'next_region':region(next_e), 'launch_after_gap_start_us':None if launch is None else launch['ts']-a,
                         'stream':next_e.get('args', {}).get('stream')})
    api = defaultdict(lambda: {'count':0, 'sum_us':0})
    for e in runtime.values():
        v=api[e['name']]; v['count']+=1; v['sum_us']+=e['dur']
    return {'gpu_span_ms':(stop-start)/1000, 'gpu_busy_union_ms':duration(occupied)/1000,
            'gpu_no_activity_ms':((stop-start)-duration(occupied))/1000,
            'compute_union_ms':duration(buckets['compute'])/1000,
            'h2d_union_ms':duration(buckets['h2d'])/1000, 'd2h_union_ms':duration(buckets['d2h'])/1000,
            'h2d_compute_overlap_ms':overlap(buckets['h2d'], buckets['compute'])/1000,
            'd2h_compute_overlap_ms':overlap(buckets['d2h'], buckets['compute'])/1000,
            'profiler_steps':[e['name'] for e in steps],
            'gpu_by_name':dict(sorted(by_name.items(), key=lambda x:x[1]['sum_us'], reverse=True)),
            'gpu_by_region':dict(by_region), 'cuda_api':dict(api),
            'largest_gpu_idle_gaps':sorted(gaps, key=lambda x:x['duration_us'], reverse=True)[:60],
            'limits':'Summed kernel/copy/API durations overlap. Copy-only time and idle gaps are not automatically removable. Profiling changes host overhead; launch-late versus dependency-wait is a diagnostic clue, not proof of a counterfactual gain.'}


def memory(data):
    results=[]
    for device, events in enumerate(data.get('device_traces', [])):
        if not events:
            continue
        current={b['address']:{'size':b['size'],'frames':b.get('frames', [])} for s in data['segments']
                 if s.get('device')==device for b in s['blocks'] if b['state']=='active_allocated'}
        # Reconstruct the initial live set backwards, including persistent blocks
        # allocated before the history ring begins. No assumption of a zero start.
        live=dict(current)
        for e in reversed(events):
            if e['action']=='alloc':live.pop(e['addr'],None)
            elif e['action']=='free_requested':live[e['addr']]={'size':e['size'],'frames':e.get('frames', [])}
        total=sum(x['size'] for x in live.values());peak=total;peak_live=dict(live);peak_index=-1
        for i,e in enumerate(events):
            if e['action']=='alloc':
                old=live.get(e['addr']);total-=old['size'] if old else 0
                live[e['addr']]={'size':e['size'],'frames':e.get('frames', [])};total+=e['size']
            elif e['action']=='free_requested':
                old=live.pop(e['addr'],None);total-=old['size'] if old else 0
            if total>peak:peak=total;peak_live=dict(live);peak_index=i
        def origin(v):
            frames=[f for f in v['frames'] if '/solution/' in f.get('filename','').replace('\\','/')]
            return (frames[0]['filename'].split('/solution/')[-1]+':'+str(frames[0]['line'])+':'+frames[0]['name']) if frames else 'library_or_unknown'
        groups=defaultdict(int)
        for v in peak_live.values():groups[origin(v)]+=v['size']
        results.append({'device':device,'trace_entries':len(events),'initial_live_mib':sum(v['size'] for v in current.values())/2**20,
                        'reconstructed_peak_mib':peak/2**20,'peak_event_index':peak_index,
                        'live_at_peak_mib_by_origin':{k:v/2**20 for k,v in sorted(groups.items(),key=lambda x:x[1],reverse=True)},
                        'largest_live_allocations':[{'size_mib':v['size']/2**20,'origin':origin(v),'frames':v['frames'][:12]}
                            for v in sorted(peak_live.values(),key=lambda x:x['size'],reverse=True)[:40]]})
    return {'devices':results,'limits':'Allocator history only: CUDA contexts/library allocations are not represented. Peak reconstruction is limited by history/stack coverage; compare to recorded max_allocated and max_reserved, not to the external headline alone.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    with zipfile.ZipFile(a.raw) as z:
        manifest=json.loads(z.read('MANIFEST.json'))
        assert set(z.namelist())==set(manifest)|{'MANIFEST.json'}
        for n,h in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
        provenance=json.loads(z.read('PROFILE_PROVENANCE.json'));assert provenance['source_unchanged'] is True
        for n,h in provenance['source_sha256'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h
        result={'source':provenance['commit'],'classification':'DIAGNOSTIC ONLY','raw_sha256':hashlib.sha256(a.raw.read_bytes()).hexdigest()}
        result['phases']=phases(json.loads(z.read('phases/PHASES.json')),json.loads(z.read('phases/PARENT.json')))
        result['timeline']=timeline(json.loads(gzip.decompress(z.read('timeline/timeline.json.gz'))))
        result['memory']={}
        for kind in ('allocated','reserved'):
            name='memory/'+kind+'-peak-snapshot.json.gz'
            result['memory'][kind]=memory(json.loads(gzip.decompress(z.read(name))))
            result['memory'][kind]['mark']=json.loads(z.read('memory/'+kind.upper()+'_PEAK_MARK.json'))
        a.out.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'phases':result['phases']['main_thread_partition_s'],
            'gpu_span_ms':result['timeline']['gpu_span_ms'],'gpu_no_activity_ms':result['timeline']['gpu_no_activity_ms'],
            'memory':{k:v['mark'] for k,v in result['memory'].items()}},indent=2))


if __name__=='__main__':main()
