"""Regenerate research_management/CHECKPOINT_INVENTORY.md (read-only over results_ft/; re-runnable).
Usage: ./torch-venv/bin/python analysis/ft/make_ckpt_inventory.py [--no-sha]"""
import re, json, hashlib, subprocess, sys, time, csv, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
RF = ROOT / "results_ft"; OUT = ROOT / "research_management/CHECKPOINT_INVENTORY.md"
SHA = "--no-sha" not in sys.argv
KEYS = ["mode", "lora-targets", "lora-r", "replicate", "loss", "head", "epochs", "seed", "lr-head", "lr-lora", "lr-trunk"]

def mb(n): return f"{n/1e6:.1f} MB" if n < 1e9 else f"{n/1e9:.2f} GB"
def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()[:16]
def argv_dict(argv):
    d, i = {}, 0
    while i < len(argv):
        if argv[i].startswith("--"):
            k = argv[i][2:]
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"): d[k] = argv[i + 1]; i += 2
            else: d[k] = "1"; i += 1
        else: i += 1
    return d
def jload(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return {}
def load_meta(p):
    import torch
    try:
        c = torch.load(p, map_location="cpu", weights_only=False)
        m = c.get("meta", {}); fs = m.get("frozen_source") or {}
        return f"{fs.get('hf_repo_id','?')}@{str(fs.get('revision','?'))[:10]}", m
    except Exception as e: return f"ERR {type(e).__name__}", {}
def du(p):
    return subprocess.run(["du", "-sb", str(p)], capture_output=True, text=True).stdout.split()[0]

L = [f"# Checkpoint inventory (generated {time.strftime('%Y-%m-%d %H:%M')} by analysis/ft/make_ckpt_inventory.py; do not hand-edit)\n",
     "Genome build hg38; all runs fine-tune/probe Borzoi (frozen_source column = pretrained trunk re-loaded from HF cache). "
     "Checkpoints hold trainable params only (+ optimizer/RNG for ckpts/ files). `results_ft/` is gitignored (`results*/`); no checkpoint is in Git.\n",
     "Reload: `analysis/ft/predict_ft.py --run <ID> --split val` rebuilds the model from `run_cmd.json` argv via `train_ft.parse()/build()` and loads `best_model.pt` key-for-key; resume training from newest `ckpts/ckpt_step*.pt` via the run's original `train_ft.py` argv.\n"]
runs = sorted((RF / "runs").iterdir(), key=lambda p: p.name)
full, prof = [], []
for r in runs:
    st = jload(r / "STATUS.json"); a = argv_dict(jload(r / "run_cmd.json").get("argv", []))
    (full if (r / "best_model.pt").exists() or (r / "stdout.log").exists() else prof).append((r, st, a))
L += ["## Training runs (results_ft/runs)\n",
      "| run | mode / arm | rep | loss | ep | seed | state | best ep | best full-val r183 | r152 | best_model.pt (size, sha256[:16]) | frozen trunk |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
extra = []
for r, st, a in full:
    ep = {}
    log = r / "stdout.log"
    if log.exists():
        for m in re.finditer(r"EPOCH (\d+) full val.*?mean r183 ([\d.]+) r152 ([\d.]+)", log.read_text(errors="ignore")):
            ep[int(m[1])] = (float(m[2]), float(m[3]))
    be = max(ep, key=lambda k: ep[k][0]) if ep else None
    arm = a.get("mode", "?") + "".join(f" {k}={a[k]}" for k in ["lora-targets", "lora-r"] + [k for k in a if "unfreeze" in k] if k in a)
    bm = r / "best_model.pt"
    if bm.exists():
        fs, meta = load_meta(bm); bmc = f"{mb(bm.stat().st_size)}, `{sha16(bm) if SHA else 'skipped'}`"
    else: fs, bmc = "n/a", "none yet"
    ck = sorted((r / "ckpts").glob("*.pt")) if (r / "ckpts").exists() else []
    if not bm.exists() and ck: fs = load_meta(ck[-1])[0]
    L.append(f"| {r.name} | {arm} | {a.get('replicate','0')} | {a.get('loss','?')} | {a.get('epochs','?')} | {a.get('seed','?')} | {st.get('state','?')} (ep {st.get('epoch','?')}, step {st.get('global_step','?')}) | "
             f"{be if be else '-'} | {ep[be][0]:.4f} | {ep[be][1]:.4f} | " .replace("| -  |", "| - |") if be else
             f"| {r.name} | {arm} | {a.get('replicate','0')} | {a.get('loss','?')} | {a.get('epochs','?')} | {a.get('seed','?')} | {st.get('state','?')} (ep {st.get('epoch','?')}, step {st.get('global_step','?')}) | - | - | - | ")
    L[-1] += f"{bmc} | {fs} |"
    extra.append((r, ck, st))
L += ["\nbest ep / r183 / r152 are the max over `EPOCH n full val` lines in each run's stdout.log (full validation set, 750 tiles; r = mean per-track Pearson r over 183 / 152 tracks). Paths: `results_ft/runs/<run>/best_model.pt`.\n",
      "### Recent resume checkpoints (results_ft/runs/<run>/ckpts/)\n", "| run | files (size) |", "|---|---|"]
for r, ck, st in extra:
    L.append(f"| {r.name} | " + ("; ".join(f"{c.name} ({mb(c.stat().st_size)})" for c in ck) or "none") + " |")
if prof:
    L += ["\n### Profile-only runs (no checkpoints; STATUS.json state)\n", "| run | state | argv (mode / key args) |", "|---|---|---|"]
    for r, st, a in prof:
        L.append(f"| {r.name} | {st.get('state','?')} | " + " ".join(f"{k}={v}" for k, v in a.items() if k != "run-id") + " |")
# cache heads
L += ["\n## Cache-based heads (results_ft/cache_exps)\n",
      "Heads trained on the frozen trunk cache (`/nobackup/.../trunk_cache`); config per id in `results_ft/cache_exps/configs/<id>.json`; val metrics from `results_ft/cache_exps/heads_val_results.csv` (column names as in that file: val_r_all183 = mean Pearson r over 183, val_r_152). "
      "Rows of the CSV wider than its header (mlp heads) are parsed from the row tail (fields -4/-3 = r183/r152; verified against FT-C83..85 rows). Config `checkpoint` field points to `/tmp/yejz1_ft/ckpt/` (node-local); the persistent copy is `results_ft/cache_exps/ckpt/`.\n",
      "| id | family | best_ep | val r183 | val r152 | ckpt (size) | sha256[:16] |", "|---|---|---|---|---|---|---|"]
res = {}
rp = RF / "cache_exps/heads_val_results.csv"
if rp.exists():
    rows = list(csv.reader(open(rp))); hdr = rows[0]
    for r_ in rows[1:]:
        if not r_: continue
        if len(r_) == len(hdr): res[r_[0]] = dict(zip(hdr, r_))
        else:  # later rows have a wider schema than the file header (mlp heads): val r183, r152, seconds, gpu_gb are the last 4 fields
            res[r_[0]] = dict(exp_id=r_[0], family=r_[1], val_r_all183=r_[-4], val_r_152=r_[-3], best_epoch=r_[-5], _wide="1")
for c in sorted((RF / "cache_exps/ckpt").glob("*.pt")):
    cfg = jload(RF / "cache_exps/configs" / f"{c.stem}.json"); row = res.get(c.stem, {})
    f = lambda k: f"{float(row[k]):.4f}" if row.get(k) not in (None, "") else "-"
    L.append(f"| {c.stem} | {cfg.get('family', row.get('family','?'))} | {cfg.get('best_epoch', row.get('best_epoch','-'))} | {f('val_r_all183')} | {f('val_r_152')} | {mb(c.stat().st_size)} | `{sha16(c) if SHA else 'skipped'}` |")
# preds / blends
L += ["\n## Validation predictions (results_ft/preds)\n", "| dir | total size | files |", "|---|---|---|"]
for d in sorted((RF / "preds").iterdir()):
    if d.is_dir():
        fl = sorted(d.iterdir()); L.append(f"| {d.name} | {mb(sum(x.stat().st_size for x in fl))} | " + ", ".join(f"{x.name} ({mb(x.stat().st_size)})" for x in fl) + " |")
L += ["\n## Blends (results_ft/blends)\n", "| path | size |", "|---|---|"]
for p in sorted((RF / "blends").rglob("*")):
    if p.is_file(): L.append(f"| {p.relative_to(RF)} | {mb(p.stat().st_size)} |")
# storage
tmp = Path("/tmp/yejz1_ft"); tc = Path("/nobackup/user/yejz1/crc_phase2/trunk_cache")
acc = subprocess.run("accre_storage 2>&1 | grep -E 'Home|Nobackup'", shell=True, capture_output=True, text=True).stdout.strip()
L += ["\n## Storage and persistence\n",
      f"- `results_ft/` total: {mb(int(du(RF)))} (du -sb). Subdirs: " + ", ".join(f"{p.name} {mb(int(du(p)))}" for p in sorted(RF.iterdir()) if p.is_dir()) + ".",
      "- `results_ft/` lives on home: persistent and backed up (FT_PLAN.md). Gitignored (`results*/`); checkpoints are NOT in Git.",
      f"- Home / nobackup usage (`accre_storage`, columns usage/quota/limit then files):\n```\n{acc}\n```"]
if tmp.exists():
    L.append("- `/tmp/yejz1_ft` is node-local, NOT persistent, not backed up: " + ", ".join(f"{p.name} {mb(int(du(p)))}" for p in sorted(tmp.iterdir()))
             + ". `pooled128` is regenerable via `analysis/ft/build_pooled.py` (needs trunk cache); `ckpt/` copies of cache heads are duplicated in `results_ft/cache_exps/ckpt/` (verify sha before relying on either).")
else: L.append("- `/tmp/yejz1_ft` absent on this node (regenerate via `analysis/ft/build_pooled.py`).")
L.append(f"- Trunk cache `{tc}`: {'present, ' + mb(int(du(tc))) if tc.exists() else 'MISSING'}; read-only input, on /nobackup (not backed up).")
OUT.write_text("\n".join(L) + "\n"); print("wrote", OUT)
