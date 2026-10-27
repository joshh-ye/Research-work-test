"""Compare saved trainable trunk tensors of the 8 ensemble members with the pretrained HF replicate; LoRA adapter (lora_B) non-zero check. CPU; writes results_ft/ablation/weight_change_check.json"""
import torch, sys, json
sys.path.insert(0, "analysis/ft")
from borzoi_pytorch import Borzoi
out = {}
for run, rep in [("FT-S03",0),("FT-R1",1),("FT-R2",2),("FT-R3",3),("FT-S04",0),("FT-L1",1),("FT-L2",2),("FT-L3",3)]:
    pre = Borzoi.from_pretrained(f"johahi/borzoi-replicate-{rep}").state_dict()
    p = torch.load(f"results_ft/runs/{run}/best_model.pt", map_location="cpu", weights_only=False)
    tr = p["trainable"]
    n_tr = sum(v.numel() for v in tr.values())
    trunk = {k: v for k, v in tr.items() if not k.startswith("head") and "lora" not in k.lower()}
    changed = []; same = 0; matched = 0; nmatch_par=0
    for k, v in trunk.items():
        kk = k[len("trunk."):] if k.startswith("trunk.") else k
        if kk in pre:
            matched += 1; nmatch_par += v.numel()
            d = (v.float() - pre[kk].float()).abs().max().item()
            rel = ((v.float()-pre[kk].float()).norm()/pre[kk].float().norm().clamp_min(1e-12)).item()
            changed.append((k, d, rel))
    lora = {k: v for k, v in tr.items() if "lora" in k.lower()}
    out[run] = dict(n_trainable_saved=n_tr, n_trunk_keys=len(trunk), n_trunk_keys_matched_pretrained=matched, trunk_params_matched=nmatch_par,
        n_changed=sum(1 for c in changed if c[1] > 0), median_rel_change=float(torch.tensor([c[2] for c in changed]).median()) if changed else None,
        max_rel_change=max([c[2] for c in changed]) if changed else None, n_lora_keys=len(lora), lora_params=sum(v.numel() for v in lora.values()),
        n_lora_B=sum(1 for k in lora if k.endswith("lora_B")), n_lora_B_nonzero=sum(1 for k,v in lora.items() if k.endswith("lora_B") and v.abs().max()>0), median_lora_B_norm=float(torch.tensor([v.float().norm() for k,v in lora.items() if k.endswith("lora_B")]).median()) if lora else None, head_params=sum(v.numel() for k,v in tr.items() if k.startswith("head")), example_keys=list(tr.keys())[:3], meta={k: p["meta"][k] for k in ["epoch","global_step","val_mean_pearson_r_full"] if k in p["meta"]})
    print(run, json.dumps(out[run]), flush=True)
json.dump(out, open("results_ft/ablation/weight_change_check.json","w"), indent=1)
