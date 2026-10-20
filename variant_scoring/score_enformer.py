"""Variant-effect scoring with pretrained Enformer (enformer_pytorch port, 5313 human tracks), hg38.
Uses torch-venv + enformer_pytorch exactly like baseline_enformer.py (EleutherAI/enformer-official-rough);
the project's tf-venv has no enformer weights/loader. SD-005 spec in variant_scoring/SCORING_SPEC.md.

Usage (GPU):
  ./torch-venv/bin/python variant_scoring/score_enformer.py --smoke
  ./torch-venv/bin/python variant_scoring/score_enformer.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))

from common import ENFORMER, load_variants  # noqa: E402
import runner  # noqa: E402
from score_borzoi import select_smoke, OUT_ROOT  # noqa: E402

N_ENF = 5313
MODEL_ID = "EleutherAI/enformer-official-rough"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--positions-per-chunk", type=int, default=300)
    ap.add_argument("--finalize-only", action="store_true")
    args = ap.parse_args()

    import torch
    from enformer_pytorch import Enformer
    device = torch.device("cuda")
    v = load_variants()
    out = Path(args.out_dir) if args.out_dir else OUT_ROOT / ("enformer_smoke" if args.smoke else "enformer")
    if args.smoke:
        v = select_smoke(v)
        print(f"[smoke] {len(v)} rows: {v['rsid'].tolist()}")
    sl = ENFORMER.agg_slice()
    pre = pd.read_csv(HERE / "track_lists" / "enformer_pretrained_crc_tracks.tsv", sep="\t")
    mask = np.zeros(N_ENF, bool); mask[pre["track_index"].to_numpy()] = True
    heads = {"pretrained": N_ENF}
    if not args.finalize_only:
        model = Enformer.from_pretrained(MODEL_ID).eval().to(device)

        def predict(batch):
            x = torch.from_numpy(batch).to(device)                                   # (B, 196608, 4) ACGT
            with torch.no_grad():
                y = model(x, head="human")                                           # (B, 896, 5313)
            assert y.shape[1] == ENFORMER.n_bins and y.shape[2] == N_ENF, y.shape
            return {"pretrained": y[:, sl, :].float().cpu().numpy()}

        t0 = time.time()
        runner.run(v, ENFORMER, predict, heads, out, args.positions_per_chunk, control_ref_eq_alt=args.smoke)
        print(f"[done] wall={time.time()-t0:.1f}s", flush=True)
    runner.finalize(v, heads, out, {"pretrained": {"crc_dnase_tfchip": mask}})


if __name__ == "__main__":
    main()
