"""Verify the 8 ensemble-member checkpoints against CHECKPOINT_INVENTORY.md (sha256[:16]); read-only. Writes research_management/finetuning/checkpoint_verification.md"""
import re, hashlib, time, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
inv = (ROOT / "research_management/CHECKPOINT_INVENTORY.md").read_text()
MEM = ["FT-S03", "FT-R1", "FT-R2", "FT-R3", "FT-S04", "FT-L1", "FT-L2", "FT-L3"]
def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()[:16]
rows = ["# Checkpoint verification (ensemble members)", f"Run {time.strftime('%Y-%m-%d %H:%M %Z')} by `analysis/ft/verify_ckpts.py`. Storage: `{ROOT}/results_ft/runs/` resolves to `{os.path.realpath(ROOT / 'results_ft')}` (ACCRE home/panfs, persistent across VM sessions; not node-local).", "",
        "| run | size | sha256[:16] now | inventory | match |", "|---|---|---|---|---|"]
ok = True
for r in MEM:
    p = ROOT / "results_ft/runs" / r / "best_model.pt"
    m = re.search(rf"^\| {r} \|.*?`([0-9a-f]{{16}})`", inv, re.M)
    s = sha16(p); ref = m.group(1) if m else "?"
    ok &= s == ref
    rows.append(f"| {r} | {p.stat().st_size/1e6:.1f} MB | `{s}` | `{ref}` | {'yes' if s == ref else 'NO'} |")
rows += ["", f"All match: {ok}"]
(ROOT / "research_management/finetuning/checkpoint_verification.md").write_text("\n".join(rows) + "\n")
print("\n".join(rows))
