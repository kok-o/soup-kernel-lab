"""CPU-only checks that the observer preserves calls, exceptions and code bytes."""
import hashlib
import importlib.util
import marshal
from pathlib import Path
import threading


def test_local_monitor():
    spec = importlib.util.spec_from_file_location('profile_stack', Path(__file__).with_name('profile_stack.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def recur(n):
        if n < 0:
            raise ValueError('expected')
        return 1 if n == 0 else recur(n-1)+1
    before = hashlib.sha256(marshal.dumps(recur.__code__)).hexdigest()
    mon = module.PhaseMonitor()
    mon.add(recur, 'recursive')
    try:
        assert recur(3) == 4
        t = threading.Thread(target=lambda: recur(1))
        t.start()
        t.join()
        try:
            recur(-1)
        except ValueError as exc:
            assert str(exc) == 'expected'
        else:
            raise AssertionError('Exception was suppressed')
    finally:
        mon.stop()
    assert len(mon.rows) == 7
    assert len({r['id'] for r in mon.rows}) == 7
    assert all('end_s' in r and r['end_s'] >= r['start_s'] for r in mon.rows)
    assert sum(bool(r.get('exception_exit')) for r in mon.rows) == 1
    assert len({r['thread'] for r in mon.rows}) == 2
    assert hashlib.sha256(marshal.dumps(recur.__code__)).hexdigest() == before


if __name__ == '__main__':
    test_local_monitor()
    print('Phase monitor: recursion, threads, exceptions and code bytes preserved.')
