"""Consolidate validation results (ridge + heads) from configs/ and per_track_val_r/ into one table; validation only."""
import json, glob, sys
import numpy as np, pandas as pd
ROOT = "/home/yejz1/experimental_research/results_ft/cache_exps"
ident = (pd.read_csv("/home/yejz1/experimental_research/exposure_audit/track_exposure.tsv", sep="\t").Borzoi_class == "identical_experiment").values
reg = pd.read_csv("/home/yejz1/experimental_research/research_management/finetuning/FT_REGISTRY.csv").set_index("exp_id")
rows = []
for f in sorted(glob.glob(f"{ROOT}/configs/FT-C*.json")):
    c = json.load(open(f)); i = f.split("/")[-1][:-5]
    try: r = np.load(f"{ROOT}/per_track_val_r/{i}.npz")["r"]
    except FileNotFoundError: continue
    rows.append(dict(exp_id=i, family=c["family"], src="+".join(c["src"]), pre_tf=c["pre_tf"], offsets=",".join(map(str, c["offsets"])), ctx="+".join(c.get("ctx", c["src"])), p=c["p"],
                     ytf=c["ytf"], backtransform=c["backtransform"], lam_mode=c.get("lam_mode", ""), lam=c.get("lam", ""), lr=c.get("lr", ""), dropout=c.get("dropout", ""),
                     best_epoch=c.get("best_epoch", ""), max_epochs=c.get("max_epochs", ""), val_r_all183=float(np.nanmean(r)), val_r_152=float(np.nanmean(r[~ident])),
                     runtime_s=reg.loc[i, "runtime_s"], gpu_mem_gb=reg.loc[i, "gpu_mem_gb"]))
d = pd.DataFrame(rows); d.to_csv(f"{ROOT}/val_results_all.csv", index=False)
print(d.sort_values("val_r_all183", ascending=False).head(25).round(4).to_string())
