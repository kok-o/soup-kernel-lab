"""Diagnostic-only phase, CPU/CUDA and allocation traces of an unchanged snapshot.

No file under solution/, harness/, task/ or tests/ is modified. These runs are
explicitly unscored; timing/profiler/snapshot overhead must never be qualified.
"""
import time
ENTRY_PERF = time.perf_counter()
import argparse
import atexit
from contextlib import contextmanager, nullcontext
import functools
import gzip
import importlib._bootstrap as bootstrap
import inspect
import itertools
import json
from pathlib import Path
import runpy
import subprocess
import sys
import threading
import types


class PhaseMonitor:
    def __init__(self):
        self.rows = []
        self.ids = itertools.count()
        self.codes = {}
        self.local = threading.local()
        self.marks = []
        self.tool = next(i for i in (2, 4, 3, 5) if sys.monitoring.get_tool(i) is None)
        m = sys.monitoring
        m.use_tool_id(self.tool, 'soup-diagnostic-phases')
        m.register_callback(self.tool, m.events.PY_START, self.enter)
        m.register_callback(self.tool, m.events.PY_RETURN, self.leave)
        m.register_callback(self.tool, m.events.PY_UNWIND, self.unwind)
        m.set_events(self.tool, m.events.PY_UNWIND)
        self.on_enter = lambda label: None
        self.on_leave = lambda label: None

    def begin(self, label):
        stack = getattr(self.local, 'stack', [])
        row = {'id': next(self.ids), 'parent': stack[-1]['id'] if stack else None,
               'name': label, 'thread': threading.get_ident(),
               'thread_name': threading.current_thread().name,
               'start_s': time.perf_counter() - ENTRY_PERF}
        self.rows.append(row)
        self.local.stack = stack + [row]
        return row

    def end(self, row, exception=False):
        row['end_s'] = time.perf_counter() - ENTRY_PERF
        if exception:
            row['exception_exit'] = True
        stack = getattr(self.local, 'stack', [])
        self.local.stack = [r for r in stack if r is not row]

    def add(self, obj, label):
        if isinstance(obj, (classmethod, staticmethod)):
            obj = obj.__func__
        if inspect.ismethod(obj):
            obj = obj.__func__
        try:
            obj = inspect.unwrap(obj)
        except ValueError:
            return
        code = getattr(obj, '__code__', obj if isinstance(obj, types.CodeType) else None)
        if code is None or code in self.codes or code.co_flags & (inspect.CO_GENERATOR | inspect.CO_COROUTINE):
            return
        self.codes[code] = label
        sys.monitoring.set_local_events(self.tool, code,
                                       sys.monitoring.events.PY_START | sys.monitoring.events.PY_RETURN)

    def enter(self, code, offset):
        label = self.codes.get(code)
        if label is None:
            return
        row = self.begin(label)
        active = getattr(self.local, 'active', {})
        active.setdefault(code, []).append(row)
        self.local.active = active
        self.on_enter(label)

    def finish(self, code, exception=False):
        active = getattr(self.local, 'active', {})
        rows = active.get(code, [])
        if rows:
            row = rows.pop()
            self.end(row, exception)
            self.on_leave(row['name'])

    def leave(self, code, offset, value):
        self.finish(code)

    def unwind(self, code, offset, exception):
        self.finish(code, True)

    def stop(self):
        m = sys.monitoring
        m.set_events(self.tool, 0)
        for code in self.codes:
            m.set_local_events(self.tool, code, 0)
        for event in (m.events.PY_START, m.events.PY_RETURN, m.events.PY_UNWIND):
            m.register_callback(self.tool, event, None)
        m.free_tool_id(self.tool)


def child(root, directory, mode, light_timeline=False):
    assert not any(n in sys.modules for n in ('torch', 'transformers', 'solution', 'soup_cli'))
    mon = PhaseMonitor()
    original_import = bootstrap._find_and_load
    original_argv = sys.argv[:]
    busy = False
    hooked = set()
    warnings = []
    timeline_active = False
    memory_started = False
    last_peak = 0
    last_reserved = 0
    profiler = None
    step_records = []
    callback_installed = False
    first_step = False
    targets = {
        'solution': [('apply', 'solution.apply')],
        'solution.runtime_compile': [('start_runtime_compile', 'compile.start'),
                                     ('_compile', 'compile.job'), ('finish_runtime_compile', 'compile.join')],
        'transformers.models.auto.auto_factory': [('_BaseAutoModelClass.from_pretrained', 'model.auto_load')],
        'transformers.models.auto.tokenization_auto': [('AutoTokenizer.from_pretrained', 'tokenizer.load')],
        'transformers.modeling_utils': [('PreTrainedModel.from_pretrained', 'model.from_pretrained'),
                                      ('convert_and_load_state_dict_in_model', 'model.load_convert')],
        'transformers.integrations.bitsandbytes': [('Bnb4bitQuantize.convert', 'model.quantize')],
        'solution.stream_nf4': [('HostNF4.capture', 'model.pin_nf4')],
        'transformers.trainer': [('Trainer.__init__', 'Trainer.init'), ('Trainer.train', 'Trainer.train'),
                                 ('Trainer._inner_training_loop', 'Trainer.inner_loop'),
                                 ('Trainer.save_model', 'save.model'), ('Trainer._save_checkpoint', 'save.checkpoint'),
                                 ('Trainer.save_state', 'save.state')],
        'peft.peft_model': [('PeftModel.save_pretrained', 'save.adapter')],
        'transformers.tokenization_utils_base': [('PreTrainedTokenizerBase.save_pretrained', 'save.tokenizer')],
        'soup_cli.cli': [('run', 'soup.run')],
        'soup_cli.trainer.sft': [('_map_text_sft_rows', 'data.format'), ('_ensure_assistant_masks', 'data.masks'),
                                 ('SFTTrainerWrapper.setup', 'soup.setup'),
                                 ('SFTTrainerWrapper._setup_transformers', 'soup.model_setup'),
                                 ('SFTTrainerWrapper.train', 'soup.train')],
        'datasets.arrow_dataset': [('Dataset.from_list', 'data.from_list'), ('Dataset.map', 'data.map')],
        'datasets.fingerprint': [('update_fingerprint', 'data.fingerprint'), ('Hasher.hash', 'data.hash')],
        'common': [('write_json', 'harness.write_json')],
    }

    def memory_checkpoint(label, force=False):
        nonlocal memory_started, last_peak, last_reserved
        if mode != 'memory':
            return
        import torch
        if not memory_started:
            torch.cuda.memory._record_memory_history(enabled='all', context='alloc', stacks='python', max_entries=150000)
            memory_started = True
        row = {'label': label, 's': time.perf_counter() - ENTRY_PERF,
               'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved(),
               'max_allocated': torch.cuda.max_memory_allocated(), 'max_reserved': torch.cuda.max_memory_reserved()}
        step_records.append(row)
        allocated_changed = force or row['max_allocated'] > last_peak
        reserved_changed = force or row['max_reserved'] > last_reserved
        if allocated_changed or reserved_changed:
            last_peak = row['max_allocated']
            last_reserved = row['max_reserved']
            before = time.perf_counter()
            snapshot = torch.cuda.memory._snapshot()
            for kind, changed in [('allocated', allocated_changed), ('reserved', reserved_changed)]:
                if changed:
                    with gzip.open(directory / (kind+'-peak-snapshot.json.gz'), 'wt', encoding='utf-8') as f:
                        json.dump(snapshot, f)
                    (directory / (kind.upper()+'_PEAK_MARK.json')).write_text(json.dumps(row, indent=2))
            mon.marks.append({'name': 'snapshot.export', 's': before-ENTRY_PERF,
                              'duration_s': time.perf_counter()-before, 'label': label})

    def range_context(label):
        if mode != 'timeline' or not timeline_active:
            return nullcontext()
        import torch
        return torch.profiler.record_function(label)

    def wrap_range(owner, attr, label):
        key = (id(owner), attr)
        if key in hooked or not hasattr(owner, attr):
            return
        descriptor = inspect.getattr_static(owner, attr)
        kind = staticmethod if isinstance(descriptor, staticmethod) else classmethod if isinstance(descriptor, classmethod) else None
        original = descriptor.__func__ if kind else getattr(owner, attr)
        @functools.wraps(original)
        def marked(*args, **kwargs):
            with range_context(label):
                return original(*args, **kwargs)
        setattr(owner, attr, kind(marked) if kind else marked)
        hooked.add(key)

    def install_timeline_ranges():
        if mode != 'timeline':
            return
        module = sys.modules.get('solution.stream_nf4')
        if module is not None and hasattr(module, 'NF4Slots'):
            key = (id(module.NF4Slots), 'acquire')
            if key not in hooked:
                original = module.NF4Slots.acquire
                @contextmanager
                @functools.wraps(original)
                def acquire(self, host):
                    with range_context('NF4.weight_lease'):
                        with original(self, host) as result:
                            yield result
                module.NF4Slots.acquire = acquire
                hooked.add(key)
            wrap_range(module, 'streamed_linear', 'NF4.forward')
        module = sys.modules.get('solution.activation_offload')
        if module is not None and hasattr(module, 'offload_packed_hooks'):
            wrap_range(module, 'prefetch_layer', 'Activation.prefetch')
            key = (id(module), 'offload_packed_hooks')
            if key not in hooked:
                original_hooks = module.offload_packed_hooks
                @functools.wraps(original_hooks)
                def hooks(*args, **kwargs):
                    pack, unpack = original_hooks(*args, **kwargs)
                    @functools.wraps(pack)
                    def p(*a, **k):
                        with range_context('Activation.pack_offload'):
                            return pack(*a, **k)
                    @functools.wraps(unpack)
                    def u(*a, **k):
                        with range_context('Activation.restore'):
                            return unpack(*a, **k)
                    return p, u
                module.offload_packed_hooks = hooks
                hooked.add(key)
        module = sys.modules.get('solution.host_head')
        if module is not None and hasattr(module, 'HostHeadFunction'):
            wrap_range(module.HostHeadFunction, 'forward', 'Head.forward')
            wrap_range(module.HostHeadFunction, 'backward', 'Head.backward')
        module = sys.modules.get('solution.hosthead_stats')
        if module is not None and hasattr(module, 'stream_head_stats'):
            wrap_range(module, 'stream_head_stats', 'Head.statistics')

    def install_callback():
        nonlocal callback_installed
        if mode == 'phases' or callback_installed:
            return
        module = sys.modules.get('transformers.trainer')
        if module is None or not hasattr(module, 'Trainer'):
            return
        import torch
        from transformers import TrainerCallback
        class DiagnosticCallback(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                nonlocal profiler
                if mode == 'timeline':
                    def ready(p):
                        started = time.perf_counter()
                        p.export_chrome_trace(str(directory / 'timeline.json.gz'))
                        (directory / 'profiler-table.txt').write_text(p.key_averages().table(sort_by='self_cuda_time_total', row_limit=150))
                        mon.marks.append({'name': 'profiler.export', 's': started-ENTRY_PERF,
                                          'duration_s': time.perf_counter()-started})
                    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        schedule=torch.profiler.schedule(wait=2, warmup=1, active=3, repeat=1),
                        on_trace_ready=ready, record_shapes=not light_timeline,
                        with_stack=not light_timeline, profile_memory=not light_timeline)
                    profiler.start()
                memory_checkpoint('train.begin')

            def on_step_begin(self, args, state, control, **kwargs):
                nonlocal timeline_active
                timeline_active = mode == 'timeline' and 3 <= state.global_step < 6

            def on_step_end(self, args, state, control, **kwargs):
                nonlocal timeline_active
                if profiler is not None and state.global_step <= 6:
                    profiler.step()
                timeline_active = False
                memory_checkpoint('step.end.' + str(state.global_step))

            def on_train_end(self, args, state, control, **kwargs):
                if profiler is not None:
                    profiler.stop()
                memory_checkpoint('train.end')
        original = module.Trainer.__init__
        @functools.wraps(original)
        def init(self, *a, **kw):
            original(self, *a, **kw)
            self.add_callback(DiagnosticCallback())
        module.Trainer.__init__ = init
        callback_installed = True

    def register_nested(code):
        for item in code.co_consts:
            if isinstance(item, types.CodeType):
                if item.co_name == 'on_step_begin':
                    mon.add(item, 'harness.step.begin')
                elif item.co_name == 'on_step_end':
                    mon.add(item, 'harness.step.end')
                register_nested(item)

    def install():
        nonlocal busy
        if busy:
            return
        busy = True
        try:
            for name, entries in targets.items():
                module = sys.modules.get(name)
                if module is None or getattr(getattr(module, '__spec__', None), '_initializing', False):
                    continue
                for path, label in entries:
                    obj = module
                    for part in path.split('.'):
                        obj = vars(obj).get(part) if hasattr(obj, '__dict__') else None
                        if obj is None:
                            break
                    if obj is not None:
                        mon.add(obj, label)
            main = sys.modules.get('__main__')
            if main is not None and hasattr(main, 'main') and getattr(main, '__file__', None) == str(root / 'harness/_train.py'):
                register_nested(main.main.__code__)
            install_timeline_ranges()
            install_callback()
        except Exception as exc:
            if len(warnings) < 20:
                warnings.append(type(exc).__name__ + ': ' + str(exc))
        finally:
            busy = False

    def importing(name, import_):
        capture = name in ('torch', 'transformers', 'bitsandbytes', 'trl', 'datasets', 'peft', 'soup_cli.cli', 'triton') and name not in sys.modules
        row = mon.begin('import.' + name) if capture else None
        try:
            return original_import(name, import_)
        finally:
            if row is not None:
                mon.end(row)
            if name.partition('.')[0] in ('torch', 'transformers', 'bitsandbytes', 'trl', 'datasets', 'peft', 'soup_cli', 'solution', 'triton', 'common'):
                install()

    def enter(label):
        nonlocal first_step
        if label == 'solution.apply' and mode == 'memory':
            memory_checkpoint('apply.begin')
        if label == 'harness.step.begin' and not first_step:
            first_step = True
            bootstrap._find_and_load = original_import
            mon.marks.append({'name': 'import_observer_disabled_before_first_step', 's': time.perf_counter()-ENTRY_PERF})

    def leave(label):
        if label == 'solution.apply':
            install()
            memory_checkpoint('apply.end')
        if label == 'model.auto_load':
            memory_checkpoint('model.loaded')

    mon.on_enter, mon.on_leave = enter, leave
    def flush(stage):
        data = {'classification': 'DIAGNOSTIC ONLY; overhead included, not a qualified benchmark',
                'mode': mode, 'light_timeline': light_timeline,
                'entry_perf': ENTRY_PERF, 'last_perf': time.perf_counter(),
                'stage': stage, 'spans': mon.rows, 'marks': mon.marks, 'warnings': warnings,
                'monitored_codes': sorted(set(mon.codes.values())), 'memory_marks': step_records,
                'original_argv': original_argv, 'source_root': str(root),
                'limitations': 'Nested and concurrent spans overlap. Profile shapes/stacks retain references and perturb memory. Snapshot history has a bounded ring. No diagnostic time/peak is a qualification result.'}
        (directory / 'PHASES.json').write_text(json.dumps(data, indent=2) + '\n')
    atexit.register(lambda: flush('atexit'))
    sys.path.insert(0, str(root))
    sys.path[0] = str(root / 'harness')
    sys.argv = [str(root / 'harness/_train.py'), '--config', str(directory / 'task.yaml'),
                '--metrics', str(directory / 'metrics.json'), '--solution']
    bootstrap._find_and_load = importing
    rc = 0
    row = mon.begin('harness.runpy')
    try:
        runpy.run_path(str(root / 'harness/_train.py'), run_name='__main__')
    except SystemExit as exc:
        rc = 0 if exc.code is None else exc.code if isinstance(exc.code, int) else 1
    finally:
        mon.end(row)
        bootstrap._find_and_load = original_import
        mon.stop()
        flush('harness.return')
    return rc


def parent(source, directory, mode, light_timeline=False):
    from hashlib import sha256
    sys.path.insert(0, str(source / 'harness'))
    from common import OFFLINE_ENV, run_logged
    from run_task import make_config
    directory.mkdir(parents=True, exist_ok=False)
    make_config(directory, source / 'data/train.jsonl', 1)
    env = dict(OFFLINE_ENV, SOUP_DB_PATH=str(directory / 'experiments.db'),
               TRITON_CACHE_DIR=str(directory / 'triton_cache'), TORCHINDUCTOR_CACHE_DIR=str(directory / 'inductor_cache'))
    assert not Path(env['TRITON_CACHE_DIR']).exists()
    before = time.perf_counter()
    child_args = [sys.executable, str(Path(__file__).resolve()), '--child', '--source', str(source),
                  '--out', str(directory), '--mode', mode]
    if light_timeline:
        child_args.append('--light-timeline')
    result = run_logged(child_args, directory / 'train.log', 900, env)
    after = time.perf_counter()
    phases = json.loads((directory / 'PHASES.json').read_text()) if (directory / 'PHASES.json').exists() else {}
    metrics = json.loads((directory / 'metrics.json').read_text()) if (directory / 'metrics.json').exists() else {}
    report = {'classification': 'DIAGNOSTIC ONLY', 'mode': mode, 'parent_start_perf': before,
              'parent_end_perf': after, 'external_diagnostic': dict(zip(('exit_code', 'wall_s', 'peak_mib', 'idle_mib'), result)),
              'child_atexit_to_parent_return_s': after-phases.get('last_perf', after), 'metrics': metrics,
              'observer_sha256': sha256(Path(__file__).read_bytes()).hexdigest()}
    (directory / 'PARENT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'mode': mode, 'exit_code': result[0], 'steps': metrics.get('steps'),
                      'unscored_wall_s': result[1], 'unscored_peak_mib': result[2], 'warnings': phases.get('warnings')}), flush=True)
    assert result[0] == 0 and metrics.get('steps') == 64, 'Diagnostic child failed; inspect train.log'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=('phases', 'timeline', 'memory'), required=True)
    p.add_argument('--child', action='store_true')
    p.add_argument('--light-timeline', action='store_true')
    args = p.parse_args()
    if args.child:
        sys.exit(child(args.source.resolve(), args.out.resolve(), args.mode, args.light_timeline))
    parent(args.source.resolve(), args.out.resolve(), args.mode, args.light_timeline)


if __name__ == '__main__':
    main()
