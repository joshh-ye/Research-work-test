"""Model-agnostic, torch-free helpers for SD-005 variant scoring (hg38, 1-based input positions).

Conventions (frozen in SCORING_SPEC.md):
  * input POS is 1-based hg38; 0-based index p0 = POS - 1.
  * model window of length L is [p0 - L//2, p0 - L//2 + L) so the variant base sits at window index L//2
    (the same "centre = interval midpoint" convention as borzoi_code/dataset.py and the baseline scripts).
  * windows that run off a chromosome are N-padded (variant stays at L//2); no shifting.
  * ALT sequence = REF window with the single base at L//2 replaced; REF is asserted against the FASTA.
  * one-hot channel order ACGT, N = all-zero; reverse complement = reverse positions AND flip channels.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyfaidx

ROOT = Path(__file__).resolve().parents[1]
HG38_FASTA = ROOT / "borzoi_data" / "hg38" / "hg38.ml.fa"
VARIANT_FILE = ROOT / "alphagenome_variant_effects" / "input" / "S6_CCV_ChrPosRefAlt_rsid.tsv"

PSEUDOCOUNT = 1.0          # added to the aggregated (summed) REF and ALT signal, model output units
AGG_BP = 512               # aggregation window in bp (closest whole-bin window to 501 bp), [p0-256, p0+256)

_LUT = np.full(256, -1, dtype=np.int8)
for _i, _b in enumerate(b"ACGT"):
    _LUT[_b] = _i
    _LUT[_b + 32] = _i     # lower case


@dataclass(frozen=True)
class Geometry:
    name: str
    seq_len: int           # model input length (bp)
    n_bins: int            # model output bins
    bin_bp: int            # bp per output bin

    @property
    def out_span_start(self) -> int:       # window index of first output bin (output is centred)
        return (self.seq_len - self.n_bins * self.bin_bp) // 2

    def agg_slice(self, agg_bp: int = AGG_BP) -> slice:
        """Output-bin slice covering [centre-agg_bp/2, centre+agg_bp/2) with centre = window index L//2."""
        c = self.seq_len // 2
        lo = c - agg_bp // 2 - self.out_span_start
        hi = c + agg_bp // 2 - self.out_span_start
        assert lo % self.bin_bp == 0 and hi % self.bin_bp == 0 and 0 <= lo < hi <= self.n_bins * self.bin_bp
        return slice(lo // self.bin_bp, hi // self.bin_bp)


BORZOI = Geometry("borzoi", 524_288, 6144, 32)
ENFORMER = Geometry("enformer", 196_608, 896, 128)


def load_variants(path: Path = VARIANT_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    parts = df.iloc[:, 0].str.split(":", expand=True)
    out = pd.DataFrame({
        "row_id": np.arange(len(df)),
        "input_variant": df.iloc[:, 0],
        "rsid": df.iloc[:, 1],
        "chrom": "chr" + parts[0].astype(str),
        "pos1": parts[1].astype(int),
        "ref": parts[2].str.upper(),
        "alt": parts[3].str.upper(),
    })
    assert out["ref"].str.len().eq(1).all() and out["alt"].str.len().eq(1).all()
    assert (out["ref"] != out["alt"]).all()
    return out


def one_hot(seq: str) -> np.ndarray:
    """(L, 4) float32, ACGT, N/other = zeros. Equivalent to borzoi_code.sequence_utils.one_hot_encode (tested)."""
    idx = _LUT[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
    out = np.zeros((len(seq), 4), dtype=np.float32)
    ok = idx >= 0
    out[np.nonzero(ok)[0], idx[ok]] = 1.0
    return out


def rc_onehot(x: np.ndarray) -> np.ndarray:
    """Reverse complement of a (..., L, 4) ACGT one-hot array."""
    return np.ascontiguousarray(x[..., ::-1, ::-1])


class Fasta:
    def __init__(self, path: Path = HG38_FASTA):
        self.f = pyfaidx.Fasta(str(path), as_raw=True)

    def chrom_len(self, chrom: str) -> int:
        return len(self.f[chrom])

    def base(self, chrom: str, pos1: int) -> str:
        """Reference base at 1-based pos1 (0-based index pos1-1)."""
        return str(self.f[chrom][pos1 - 1:pos1]).upper()

    def window(self, chrom: str, pos1: int, seq_len: int):
        """REF window string with the variant base at index seq_len//2. N-pads beyond chromosome ends.
        Returns (seq, n_pad_left, n_pad_right)."""
        p0 = pos1 - 1
        start = p0 - seq_len // 2
        end = start + seq_len
        clen = self.chrom_len(chrom)
        pl, pr = max(0, -start), max(0, end - clen)
        s = str(self.f[chrom][max(0, start):min(clen, end)]).upper()
        seq = "N" * pl + s + "N" * pr
        assert len(seq) == seq_len
        return seq, pl, pr


def ref_window_onehot(fa: Fasta, chrom: str, pos1: int, ref: str, geom: Geometry):
    """One-hot REF window; asserts the stated REF equals the FASTA base at the window centre.
    Returns (onehot (L,4), n_pad_left, n_pad_right, n_N_in_agg_window)."""
    seq, pl, pr = fa.window(chrom, pos1, geom.seq_len)
    c = geom.seq_len // 2
    if seq[c] != ref.upper():
        raise AssertionError(f"REF mismatch {chrom}:{pos1} stated {ref} fasta {seq[c]}")
    assert seq[c] == fa.base(chrom, pos1)
    agg = seq[c - AGG_BP // 2: c + AGG_BP // 2]
    return one_hot(seq), pl, pr, agg.count("N")


def alt_onehot(ref_oh: np.ndarray, alt: str, geom: Geometry, ref: str | None = None) -> np.ndarray:
    """Copy of the REF one-hot with the centre base replaced by ALT."""
    c = geom.seq_len // 2
    out = ref_oh.copy()
    if ref is not None:
        assert out[c].argmax() == "ACGT".index(ref.upper()) and out[c].sum() == 1
    out[c] = 0.0
    out[c, "ACGT".index(alt.upper())] = 1.0
    return out


def rc_average(predict_fn, oh: np.ndarray, sl: slice) -> np.ndarray:
    """Forward + reverse-complement average of predictions on the output bins `sl`.

    predict_fn: (B, L, 4) float32 numpy -> (B, n_bins_total, n_tracks) numpy, OR already restricted to `sl`
    when called through score code (see score_*.py, which restrict inside predict_fn and pass sl=slice(None)).
    The slice must be symmetric about the output centre so that the RC bins map onto the same bins reversed.
    """
    batch = np.stack([oh, rc_onehot(oh)], axis=0)
    p = predict_fn(batch)                      # (2, nb, T)
    p = p[:, sl, :]
    return 0.5 * (p[0] + p[1][::-1])           # (nb_sel, T)


def agg_sum(pred_bins: np.ndarray) -> np.ndarray:
    """(nb_sel, T) -> (T,) sum over the aggregation bins."""
    return pred_bins.sum(axis=0, dtype=np.float64).astype(np.float32)


def signed_log2fc(ref_sum: np.ndarray, alt_sum: np.ndarray, pc: float = PSEUDOCOUNT) -> np.ndarray:
    return (np.log2(alt_sum.astype(np.float64) + pc) - np.log2(ref_sum.astype(np.float64) + pc)).astype(np.float32)


def summarize(lfc: np.ndarray, track_mask: np.ndarray) -> dict:
    """lfc: (n_rows, T) signed log2FC; track_mask: bool (T,). SD-005 summaries for one track set."""
    x = lfc[:, track_mask]
    a = np.abs(x)
    am = a.argmax(axis=1)
    r = np.arange(len(x))
    return {
        "primary_max_abs_log2fc": a[r, am],
        "signed_at_argmax": x[r, am],
        "argmax_track_pos": np.flatnonzero(track_mask)[am],     # index into the full model track axis
        "mean_signed_log2fc": x.mean(axis=1),
        "mean_abs_log2fc": a.mean(axis=1),
    }


def chunk_positions(variants: pd.DataFrame, positions_per_chunk: int = 300):
    """Deterministic chunking by unique (chrom,pos1) in first-appearance order. Yields (k, row_ids)."""
    key = variants["chrom"] + ":" + variants["pos1"].astype(str)
    order = pd.unique(key)
    pos_chunk = {p: i // positions_per_chunk for i, p in enumerate(order)}
    ck = key.map(pos_chunk)
    for k in sorted(ck.unique()):
        yield int(k), variants.index[ck == k].to_numpy()
