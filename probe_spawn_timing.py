"""
Trace the spawn worker boot sequence to find which phase takes >5s.
This tells us whether the crash is during import, BigWig open, or data load.
"""
import sys, os, time, glob, multiprocessing as mp
os.environ.setdefault('HF_HOME', './hf_cache')
sys.path.insert(0, 'borzoi_code')


def worker_probe(result_q):
    t0 = time.time()
    result_q.put(('process_started', round(time.time() - t0, 2)))
    try:
        import torch
        result_q.put(('import_torch', round(time.time() - t0, 2)))
        import pyBigWig
        result_q.put(('import_pyBigWig', round(time.time() - t0, 2)))
        import pyfaidx
        result_q.put(('import_pyfaidx', round(time.time() - t0, 2)))

        bw_files = sorted(glob.glob('borzoi_data/CRC_TFs_bw/*.bw'))
        handles = [pyBigWig.open(f) for f in bw_files]
        result_q.put(('opened_all_bigwigs', round(time.time() - t0, 2), len(handles)))

        vals = handles[0].stats('chr1', 196608, 327680, type='mean', nBins=4096)
        result_q.put(('stats_chr1_ok', round(time.time() - t0, 2)))

        for h in handles:
            h.close()
        result_q.put(('closed_all', round(time.time() - t0, 2)))
    except Exception as e:
        result_q.put(('error', type(e).__name__, str(e)))


if __name__ == '__main__':
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=worker_probe, args=(q,))
    t_launch = time.time()
    p.start()
    print(f"Worker PID {p.pid} spawned.  PyTorch watchdog timeout = 5s\n")
    print(f"{'event':<30}  {'elapsed_s':>10}")
    print("-" * 45)

    events = []
    deadline = time.time() + 120
    last_event_time = time.time()
    while p.is_alive() and time.time() < deadline:
        try:
            ev = q.get(timeout=2)
            elapsed = round(time.time() - t_launch, 2)
            label = ev[0]
            extra = '  ' + str(ev[1:]) if len(ev) > 1 else ''
            print(f"{label:<30}  {elapsed:>10.2f}s{extra}")
        except Exception:
            pass

    p.join(timeout=5)
    total = round(time.time() - t_launch, 2)
    print(f"\nWorker alive={p.is_alive()}  exitcode={p.exitcode}  total={total}s")
    print(f"\nKey: PyTorch kills workers only if is_alive()==False after 5s timeout.")
    print(f"     If any phase above takes >5s AND the worker is killed (OOM/signal),")
    print(f"     that phase is where the crash happens.")
