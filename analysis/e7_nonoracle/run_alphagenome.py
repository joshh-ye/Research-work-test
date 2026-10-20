"""E7: AlphaGenome (hosted API) per-tile cross sufficient statistics (183 targets x 1,617 CHIP_TF tracks), shared grid.
Same call as baseline_alphagenome.py (predict_interval, 524,288 bp tile, CHIP_TF, ontology_terms=None, crop central 114,688 bp, pool to 128 bp).
Sequential (1 request at a time, same rate as the existing pipeline), checkpoint per tile, <=3 retries per tile, auth/quota errors -> BLOCKED stop.
The API key is read from alphagenome_variant_effects/.alphagenome_api_key (or $ALPHAGENOME_API_KEY) and never printed.
Usage (ag-venv): run_alphagenome.py --split val|test [--limit N]"""
import argparse, os, sys, time, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from e7lib import *
from enformer_baseline_utils import targets_to_enformer_grid
from baseline_alphagenome import to_common_grid, TILE_LEN

ap = argparse.ArgumentParser(); ap.add_argument("--split", required=True, choices=["val", "test"]); ap.add_argument("--limit", type=int)
ap.add_argument("--max-retries", type=int, default=3); args = ap.parse_args()
OUT = ROOT / "results_phase2/e7_nonoracle/alphagenome"; TD = OUT / f"tiles_{args.split}"; TD.mkdir(parents=True, exist_ok=True)
key = os.environ.get("ALPHAGENOME_API_KEY", "").strip()
kf = ROOT / "alphagenome_variant_effects/.alphagenome_api_key"
if not key and kf.exists(): key = kf.read_text().strip()
if not key: sys.exit("BLOCKED: no API key available (existing key file missing)")
from alphagenome.models import dna_client
from alphagenome.data import genome
client = dna_client.create(key)
ref_tracks = pd.read_csv(ROOT / "results_baseline_alphagenome/alphagenome_tracks_test.csv")
iv, tg, tiles = load_split(args.split); N = args.limit or len(iv)
FATAL = ("PERMISSION_DENIED", "UNAUTHENTICATED", "RESOURCE_EXHAUSTED", "API key", "quota", "Quota")
consec = 0; t0 = time.time(); fails = []
for i in range(N):
    f = TD / f"tile_{i:04d}.npz"
    if f.exists(): continue
    assert iv[i].end - iv[i].start == TILE_LEN
    out = None
    for att in range(args.max_retries):
        try:
            out = client.predict_interval(interval=genome.Interval(chromosome=iv[i].chrom, start=iv[i].start, end=iv[i].end),
                                          organism=dna_client.Organism.HOMO_SAPIENS, requested_outputs=[dna_client.OutputType.CHIP_TF], ontology_terms=None)
            break
        except Exception as e:
            msg = f"{type(e).__name__}: {str(e)[:300]}"
            if any(s in msg for s in FATAL):
                print(f"BLOCKED: fatal API error at tile {i}: {msg}", flush=True); (OUT / "BLOCKED.txt").write_text(msg); sys.exit(2)
            fails.append((i, att, msg)); time.sleep(5 * 2 ** att)
    if out is None:
        consec += 1; print(f"tile {i} failed after retries: {fails[-1][2]}", flush=True)
        if consec >= 5: (OUT / "BLOCKED.txt").write_text("5 consecutive tile failures: " + fails[-1][2]); sys.exit("BLOCKED: 5 consecutive tile failures")
        continue
    consec = 0
    td = out.chip_tf
    assert td.values.shape[-1] == len(ref_tracks) and (td.metadata.reset_index(drop=True)["name"].values == ref_tracks["name"].values).all(), "track order/metadata changed"
    p = to_common_grid(np.asarray(td.values), int(td.resolution))
    st = tile_cross_stats(targets_to_enformer_grid(tg[i]), p)
    np.savez(f.with_suffix(".tmp.npz"), **st); os.replace(f.with_suffix(".tmp.npz"), f)
    if (i + 1) % 10 == 0: print(f"{i+1}/{N} {time.time()-t0:.0f}s retries_so_far={len(fails)}", flush=True)
pd.DataFrame(fails, columns=["tile", "attempt", "error"]).to_csv(OUT / f"retry_log_{args.split}.csv", index=False)
fs = [np.load(TD / f"tile_{i:04d}.npz") for i in range(N)] if all((TD / f"tile_{i:04d}.npz").exists() for i in range(N)) else None
if fs is None: sys.exit("incomplete: some tiles missing (rerun to resume)")
S = {k: np.stack([z[k] for z in fs]) for k in ["n", "sT", "sT2", "sP", "sP2", "sTP"]}
np.savez(OUT / (f"tile_stats_{args.split}.npz" if not args.limit else f"tile_stats_{args.split}_limit{N}.npz"), **S, chrom=tiles.chrom.values[:N], start=tiles.start.values[:N],
         tile_zero_signal=tiles.zero_signal_all_tracks.values[:N])
print("saved", flush=True)
