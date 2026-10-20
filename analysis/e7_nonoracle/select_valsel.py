"""Validation-only selection (SELECTION_RULE.md items 3 and 5). Reads ONLY validation tile statistics. Usage: select_valsel.py enformer|alphagenome"""
import sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from e7lib import *

def candidates(model):
    if model == "enformer":
        inc = pd.read_csv(ROOT / "results_baseline_enformer/included_enformer_tracks.csv")
        meta = inc.rename(columns={"identifier": "track_id"})[["track_index", "track_id", "description"]].reset_index(drop=True)
        return meta, np.tile(identical_candidates_enformer(inc), (183, 1))
    ag = pd.read_csv(ROOT / "results_baseline_alphagenome/alphagenome_tracks_test.csv")
    meta = pd.DataFrame(dict(track_index=np.arange(len(ag)), track_id=ag["name"], description=ag.transcription_factor + " (" + ag.biosample_name + ")"))
    return meta, identical_mask_alphagenome(ag)

def main(model):
    out = ROOT / f"results_phase2/e7_nonoracle/{model}"
    z = np.load(out / "tile_stats_val.npz")
    n = z["n"].sum(); r = pearson_matrix(n, z["sT"].sum(0), z["sT2"].sum(0), z["sP"].sum(0), z["sP2"].sum(0), z["sTP"].sum(0))
    meta, ident = candidates(model); assert r.shape[1] == len(meta) == ident.shape[1]
    ex = pd.read_csv(ROOT / "exposure_audit/track_exposure.tsv", sep="\t")
    sel = select_argmax(r); sel_x = select_argmax(r, ~ident)
    ar = np.arange(183)
    df = pd.DataFrame(dict(track_idx=ar, TF=ex.TF, cell_line=ex.cell_line,
        valsel_cand_idx=sel, valsel_track_id=meta.track_id.values[sel], valsel_description=meta.description.values[sel], valsel_val_pearson_r=r[ar, sel],
        valsel_excl_ident_cand_idx=sel_x, valsel_excl_ident_track_id=meta.track_id.values[sel_x], valsel_excl_ident_description=meta.description.values[sel_x],
        valsel_excl_ident_val_pearson_r=r[ar, sel_x], valsel_selected_is_identical_candidate=ident[ar, sel], n_identical_candidates_excluded=ident.sum(1)))
    df.to_csv(out / "selection_valsel.csv", index=False)
    np.save(out / "val_pearson_matrix.npy", r)
    print(model, "mean val r valsel", np.nanmean(r[ar, sel]), "excl-identical", np.nanmean(r[ar, sel_x]),
          "n targets whose valsel pick is an identical-experiment candidate:", int(ident[ar, sel].sum()), "n ident cands (row0):", int(ident[0].sum()))
    print("selection_valsel.csv sha256", hashlib.sha256((out / "selection_valsel.csv").read_bytes()).hexdigest())

if __name__ == "__main__":
    main(sys.argv[1])
