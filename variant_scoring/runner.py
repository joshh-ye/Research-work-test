"""Resumable, chunk-checkpointed REF/ALT scoring loop shared by score_borzoi.py and score_enformer.py.

predict_fn(batch (2, L, 4) float32 [forward, reverse-complement]) -> {head: (2, n_agg_bins, T) float32}
restricted to the aggregation bins (the slice is symmetric about the output centre, so the RC output is
flipped along the bin axis and averaged with the forward output).

Per chunk (300 unique positions, ~530 rows) one .npz is written atomically; reruns skip existing chunks.
finalize() merges chunks into results arrays + summary TSVs (signed log2FC, SD-005).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import (Fasta, Geometry, alt_onehot, chunk_positions, ref_window_onehot, signed_log2fc, summarize,
                    PSEUDOCOUNT)


def validate_refs(variants: pd.DataFrame, fa: Fasta, out_dir: Path) -> pd.DataFrame:
    """Check REF==FASTA for every row (1-based pos -> 0-based index pos-1). Raise on any mismatch."""
    fasta_ref = [fa.base(c, p) for c, p in zip(variants["chrom"], variants["pos1"])]
    chk = variants[["row_id", "input_variant", "chrom", "pos1", "ref"]].copy()
    chk["fasta_ref"] = fasta_ref
    chk["ref_match"] = chk["ref"] == chk["fasta_ref"]
    out_dir.mkdir(parents=True, exist_ok=True)
    chk.to_csv(out_dir / "ref_check.tsv", sep="\t", index=False)
    n_bad = int((~chk["ref_match"]).sum())
    print(f"[ref-check] rows={len(chk)} mismatches={n_bad}", flush=True)
    if n_bad:
        raise AssertionError(f"{n_bad} REF mismatches; see {out_dir/'ref_check.tsv'}")
    return chk


def run(variants: pd.DataFrame, geom: Geometry, predict_fn, heads: dict, out_dir: Path,
        positions_per_chunk: int = 300, control_ref_eq_alt: bool = False, tag: str = ""):
    out_dir = Path(out_dir)
    chunk_dir = out_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    fa = Fasta()
    validate_refs(variants, fa, out_dir)
    chunks = list(chunk_positions(variants, positions_per_chunk))
    n_total = len(variants)
    done_rows, t0 = 0, time.time()
    for k, idx in chunks:
        f = chunk_dir / f"chunk_{k:05d}.npz"
        if f.exists():
            done_rows += len(idx)
            continue
        sub = variants.loc[idx]
        pos_key = (sub["chrom"] + ":" + sub["pos1"].astype(str)).to_numpy()
        upos = list(dict.fromkeys(pos_key))
        refs = {h: np.zeros((len(upos), T), np.float32) for h, T in heads.items()}
        alts = {h: np.zeros((len(sub), T), np.float32) for h, T in heads.items()}
        pad_l = np.zeros(len(sub), np.int32); pad_r = np.zeros(len(sub), np.int32); nN = np.zeros(len(sub), np.int32)
        ctrl = {h: np.zeros(len(sub), np.float32) for h in heads} if control_ref_eq_alt else None
        pos_of_row = np.array([upos.index(p) for p in pos_key], np.int32)

        def score(oh):
            out = predict_fn(np.stack([oh, oh[::-1, ::-1]], 0).astype(np.float32))
            res = {}
            for h, p in out.items():
                p = np.asarray(p, dtype=np.float64)
                res[h] = (0.5 * (p[0] + p[1][::-1])).sum(0).astype(np.float32)   # RC-average, sum over bins
            return res

        for pi, pk in enumerate(upos):
            rows = [i for i in range(len(sub)) if pos_of_row[i] == pi]
            r0 = sub.iloc[rows[0]]
            ref_oh, pl, pr, n_n = ref_window_onehot(fa, r0["chrom"], int(r0["pos1"]), r0["ref"], geom)
            rs = score(ref_oh)
            for h in heads:
                refs[h][pi] = rs[h]
            for i in rows:
                r = sub.iloc[i]
                assert r["ref"] == r0["ref"]
                a = score(alt_onehot(ref_oh, r["alt"], geom, ref=r["ref"]))
                for h in heads:
                    alts[h][i] = a[h]
                pad_l[i], pad_r[i], nN[i] = pl, pr, n_n
                if ctrl is not None:      # REF==ALT control through the full pipeline
                    c = score(alt_onehot(ref_oh, r["ref"], geom, ref=r["ref"]))
                    for h in heads:
                        ctrl[h][i] = np.abs(signed_log2fc(rs[h], c[h])).max()
        payload = dict(row_ids=sub["row_id"].to_numpy(), pos_keys=np.array(upos), pos_of_row=pos_of_row,
                       n_pad_left=pad_l, n_pad_right=pad_r, n_N_in_agg=nN)
        for h in heads:
            payload[f"ref_{h}"] = refs[h]; payload[f"alt_{h}"] = alts[h]
            if ctrl is not None:
                payload[f"ctrl_{h}"] = ctrl[h]
        tmp = chunk_dir / f"chunk_{k:05d}.tmp.npz"
        np.savez(tmp, **payload)
        tmp.rename(f)
        done_rows += len(sub)
        el = time.time() - t0
        print(f"[chunk {k+1}/{len(chunks)}] rows_done={done_rows}/{n_total} elapsed={el/60:.1f} min", flush=True)
    return chunks


def finalize(variants: pd.DataFrame, heads: dict, out_dir: Path, track_sets: dict, tag: str = ""):
    """Merge chunks. track_sets: {head: {set_name: bool mask (T,)}}. Writes arrays + summary TSVs."""
    out_dir = Path(out_dir)
    files = sorted((out_dir / "chunks").glob("chunk_*.npz"))
    files = [f for f in files if ".tmp" not in f.name]
    data = [np.load(f, allow_pickle=False) for f in files]
    row_ids = np.concatenate([d["row_ids"] for d in data])
    order = np.argsort(row_ids)
    row_ids = row_ids[order]
    v = variants.set_index("row_id").loc[row_ids].reset_index()
    v["n_pad_left"] = np.concatenate([d["n_pad_left"] for d in data])[order]
    v["n_pad_right"] = np.concatenate([d["n_pad_right"] for d in data])[order]
    v["n_N_in_agg_window"] = np.concatenate([d["n_N_in_agg"] for d in data])[order]
    v["window_padded"] = (v["n_pad_left"] + v["n_pad_right"]) > 0
    # per-position REF sums, mapped to rows
    pos_keys, ref_by_head = [], {h: [] for h in heads}
    for d in data:
        for j, pk in enumerate(d["pos_keys"]):
            pos_keys.append(str(pk))
            for h in heads:
                ref_by_head[h].append(d[f"ref_{h}"][j])
    pos_index = {p: i for i, p in enumerate(pos_keys)}
    assert len(pos_index) == len(pos_keys), "a position appears in two chunks"
    v["pos_id"] = [pos_index[c + ":" + str(p)] for c, p in zip(v["chrom"], v["pos1"])]
    v.to_csv(out_dir / "variants.tsv", sep="\t", index=False)
    pd.Series(pos_keys, name="chrom_pos1").to_csv(out_dir / "positions.tsv", sep="\t", index=False)
    summ = v[["row_id", "input_variant", "rsid", "chrom", "pos1", "ref", "alt", "window_padded", "n_N_in_agg_window"]].copy()
    for h in heads:
        ref = np.stack(ref_by_head[h]).astype(np.float32)
        alt = np.concatenate([d[f"alt_{h}"] for d in data])[order]
        lfc = signed_log2fc(ref[v["pos_id"].to_numpy()], alt, PSEUDOCOUNT)
        np.save(out_dir / f"ref_sum_{h}.npy", ref)          # (n_positions, T), rows of positions.tsv
        np.save(out_dir / f"alt_sum_{h}.npy", alt)          # (n_rows, T), rows of variants.tsv
        np.save(out_dir / f"log2fc_{h}.npy", lfc)           # (n_rows, T) signed, pc=PSEUDOCOUNT
        for sname, mask in track_sets[h].items():
            s = summarize(lfc, np.asarray(mask, bool))
            for key, val in s.items():
                summ[f"{h}__{sname}__{key}"] = val
        if f"ctrl_{h}" in data[0].files:
            summ[f"{h}__ctrl_ref_eq_alt_max_abs_log2fc"] = np.concatenate([d[f"ctrl_{h}"] for d in data])[order]
    summ.to_csv(out_dir / "summary.tsv", sep="\t", index=False)
    print(f"[finalize] rows={len(v)} heads={list(heads)} -> {out_dir}", flush=True)
    return summ
