"""
Run 8 spawn workers simultaneously (same as val DataLoader config) and
measure per-worker timing for each boot phase.

If any worker takes >5s to produce its first item, PyTorch's watchdog fires.
If workers are killed (OOM/signal) rather than just slow, we see the crash.
"""
import sys, os, time, glob, multiprocessing as mp
os.environ.setdefault('HF_HOME', './hf_cache')
sys.path.insert(0, 'borzoi_code')


def worker_probe(worker_id, result_q):
    t0 = time.time()
    try:
        import torch
        result_q.put((worker_id, 'import_torch', round(time.time() - t0, 2)))
        import pyBigWig, pyfaidx
        result_q.put((worker_id, 'import_libs', round(time.time() - t0, 2)))

        bw_files = sorted(glob.glob('borzoi_data/CRC_TFs_bw/*.bw'))
        handles = [pyBigWig.open(f) for f in bw_files]
        result_q.put((worker_id, 'opened_bigwigs', round(time.time() - t0, 2), len(handles)))

        # Query chr1 — the val chromosome
        vals = handles[0].stats('chr1', 196608, 327680, type='mean', nBins=4096)
        result_q.put((worker_id, 'stats_ok', round(time.time() - t0, 2)))

        # Also open the FASTA (simulates the other lazy open)
        fasta = pyfaidx.Fasta('borzoi_data/hg38/hg38.ml.fa', as_raw=True)
        seq = str(fasta['chr1'][196608:327680])
        result_q.put((worker_id, 'fasta_read', round(time.time() - t0, 2), len(seq)))
        fasta.close()

        for h in handles:
            h.close()
        result_q.put((worker_id, 'done', round(time.time() - t0, 2)))

    except Exception as e:
        result_q.put((worker_id, 'error', type(e).__name__, str(e)[:120]))


if __name__ == '__main__':
    N = 8
    ctx = mp.get_context('spawn')
    q = ctx.Queue()

    t_launch = time.time()
    procs = [ctx.Process(target=worker_probe, args=(i, q)) for i in range(N)]
    for p in procs:
        p.start()
    pids = [p.pid for p in procs]
    print(f"Spawned {N} workers: PIDs {pids}")
    print(f"PyTorch watchdog = 5s — any worker alive after 5s is fine; killed = crash\n")
    print(f"{'wid':>4}  {'event':<22}  {'elapsed':>8}")
    print("-" * 40)

    events = []
    deadline = time.time() + 120
    while any(p.is_alive() for p in procs) and time.time() < deadline:
        try:
            ev = q.get(timeout=1)
            elapsed = round(time.time() - t_launch, 2)
            wid = ev[0]
            label = ev[1]
            extra = f"  {ev[2]}" if len(ev) > 2 else ''
            print(f"{wid:>4}  {label:<22}  {elapsed:>8.2f}s{extra}")
        except Exception:
            pass

    for p in procs:
        p.join(timeout=5)

    print("\nFinal worker status:")
    for i, p in enumerate(procs):
        print(f"  worker {i}: pid={p.pid}  alive={p.is_alive()}  exitcode={p.exitcode}")

    total = round(time.time() - t_launch, 2)
    print(f"\nTotal wall time: {total}s")
