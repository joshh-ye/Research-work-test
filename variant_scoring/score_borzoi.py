"""Variant-effect scoring with pretrained Borzoi (4 replicates, 7611 human tracks) AND Borzoi-TL
(frozen backbone embedding -> MLP head borzoi_ckpt_full/model_head_best.pt, 183 CRC TF ChIP tracks)
from ONE trunk pass per sequence. hg38, SD-005 spec in variant_scoring/SCORING_SPEC.md.

Trunk tap (identical to borzoi_code/model.py): the input of `human_head`, i.e.
`final_joined_convs(get_embs_after_crop(x))` -> (B, 1920, 6144 bins x 32 bp), replicate-averaged for the TL head.
Pretrained output = final_softplus(human_head(emb)) averaged over the 4 replicates (as baseline_borzoi_filtered.py).

Usage (GPU):
  ./torch-venv/bin/python variant_scoring/score_borzoi.py --selftest
  ./torch-venv/bin/python variant_scoring/score_borzoi.py --smoke
  ./torch-venv/bin/python variant_scoring/score_borzoi.py            # full 10,148 rows, resumable
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

from common import BORZOI, load_variants, one_hot, Fasta  # noqa: E402
import runner  # noqa: E402

N_TL = 183
N_PRE = 7611
CKPT = ROOT / "borzoi_ckpt_full" / "model_head_best.pt"
OUT_ROOT = ROOT / "results_phase2" / "variant_scores"


def build_models(device):
    import torch
    import torch.nn as nn
    from borzoi_pytorch import Borzoi
    backbones = [Borzoi.from_pretrained(f"johahi/borzoi-replicate-{i}").eval().to(device) for i in range(4)]
    for b in backbones:
        for p in b.parameters():
            p.requires_grad = False
    head = nn.Sequential(nn.Linear(1920, 1024), nn.GELU(), nn.Dropout(0.2), nn.Linear(1024, N_TL), nn.Softplus())
    head.load_state_dict(torch.load(CKPT, map_location=device))
    head.to(device).eval()
    return backbones, head


def make_predict_fn(backbones, head, device, sl: slice):
    import torch

    def predict(batch: np.ndarray):
        x = torch.from_numpy(batch).permute(0, 2, 1).contiguous().to(device)      # (B, 4, L)
        pre, emb_sum = 0.0, 0.0
        with torch.no_grad():
            for b in backbones:
                e = b.final_joined_convs(b.get_embs_after_crop(x))                  # (B, 1920, 6144)
                assert e.shape[-1] == BORZOI.n_bins, e.shape
                e = e[..., sl]
                emb_sum = emb_sum + e
                pre = pre + b.final_softplus(b.human_head(e.float()))               # (B, 7611, nb)
            pre = pre / len(backbones)
            emb = emb_sum / len(backbones)                                           # replicate-averaged embedding
            tl = head(emb.float().permute(0, 2, 1))                                  # (B, nb, 183)
        return {"pretrained": pre.permute(0, 2, 1).float().cpu().numpy(),            # (B, nb, 7611)
                "tl": tl.float().cpu().numpy()}
    return predict


def track_sets():
    pre = pd.read_csv(HERE / "track_lists" / "borzoi_pretrained_crc_tracks.tsv", sep="\t")
    m_pre = np.zeros(N_PRE, bool); m_pre[pre["track_index"].to_numpy()] = True
    tl = pd.read_csv(HERE / "track_lists" / "borzoi_tl_tracks.tsv", sep="\t")
    assert (tl["tl_head_index"].to_numpy() == np.arange(N_TL)).all()
    return {"pretrained": {"crc_dnase_tfchip": m_pre},
            "tl": {"primary152": tl["primary"].to_numpy(bool), "all183": np.ones(N_TL, bool)}}


def select_smoke(v: pd.DataFrame, n_other: int = 14) -> pd.DataFrame:
    keep = set(v.index[v["rsid"] == "rs6983267"])
    assert keep, "rs6983267 not in input"
    fa = Fasta()
    L = {c: fa.chrom_len(c) for c in v["chrom"].unique()}
    edge = v.index[(v["pos1"] - 1 - BORZOI.seq_len // 2 < 0) | (v["pos1"] - 1 + BORZOI.seq_len // 2 > v["chrom"].map(L))]
    keep |= set(edge[:3])                                   # chromosome-end (N-padded) windows
    others = v.index[~v.index.isin(keep)]
    keep |= set(others[np.linspace(0, len(others) - 1, n_other).astype(int)])
    return v.loc[sorted(keep)]


def selftest(device):
    """Check the one-pass implementation against (a) full forward of the pretrained model, (b) stored TL test preds."""
    import torch
    sys.path.insert(0, str(ROOT / "borzoi_code"))
    from genome_tiler import tile_genome
    from fasta_reader import FastaReader
    fasta_path = str(ROOT / "borzoi_data" / "hg38" / "hg38.ml.fa")
    iv = tile_genome(fasta_path)["test"][60]
    center = (iv.start + iv.end) // 2
    seq = FastaReader(fasta_path).fetch(iv.chrom, center - 262144, center + 262144)
    oh = one_hot(seq)
    backbones, head = build_models(device)
    full = slice(1024, 5120)                        # the 4096-bin TL target region
    pred = make_predict_fn(backbones, head, device, full)(oh[None])
    stored = np.load(ROOT / "results_full" / "test_preds.npy", mmap_mode="r")[60]
    d_tl = np.abs(pred["tl"][0] - stored).max()
    print(f"[selftest] TL vs results_full/test_preds.npy[60]: max|diff|={d_tl:.3e}  (stored max {stored.max():.3f})")
    with torch.no_grad():
        x = torch.from_numpy(oh).permute(1, 0)[None].to(device)
        ref = np.mean([b(x).float().cpu().numpy()[0] for b in backbones], 0)[:, full]   # (7611, 4096)
    d_pre = np.abs(pred["pretrained"][0].T - ref).max()
    print(f"[selftest] pretrained vs full forward average: max|diff|={d_pre:.3e}  (ref max {ref.max():.3f})")
    return d_tl, d_pre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="~20 variants incl. rs6983267, REF==ALT control, timing")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--positions-per-chunk", type=int, default=300)
    ap.add_argument("--finalize-only", action="store_true")
    args = ap.parse_args()

    import torch
    device = torch.device("cuda")
    if args.selftest:
        selftest(device); return

    v = load_variants()
    out = Path(args.out_dir) if args.out_dir else OUT_ROOT / ("borzoi_smoke" if args.smoke else "borzoi")
    if args.smoke:
        v = select_smoke(v)
        print(f"[smoke] {len(v)} rows, {v[['chrom','pos1']].drop_duplicates().shape[0]} positions: {v['rsid'].tolist()}")
    heads = {"pretrained": N_PRE, "tl": N_TL}
    sets = track_sets()
    if not args.finalize_only:
        backbones, head = build_models(device)
        predict = make_predict_fn(backbones, head, device, BORZOI.agg_slice())
        t0 = time.time()
        runner.run(v, BORZOI, predict, heads, out, args.positions_per_chunk, control_ref_eq_alt=args.smoke)
        print(f"[done] wall={time.time()-t0:.1f}s", flush=True)
    runner.finalize(v, heads, out, sets)


if __name__ == "__main__":
    main()
