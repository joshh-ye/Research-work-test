"""Borzoi (replicate 0) fine-tuning on the 183 CRC tracks.  hg38.  Train 4,659 / val chr1+chr8 750 tiles from
results_phase2/trunk_cache/tiles.csv.  NO TEST-SET EVALUATION IN THIS SCRIPT (test tiles are never read).

Modes: frozen (control) | partial (--unfreeze blocks:K,up,final,horiz,unet1,res) | lora (--lora-targets attn,mlp,conv1x1
--lora-r --lora-blocks) | full.  Head: mlp (1920-1024-183, GELU, dropout .2, Softplus = original) | linear (1920-183, Softplus).
Loss: poisson (original; F.poisson_nll_loss(log_input=False) mean over bins x tracks) | mse128 (MSE of 128 bp mean-pooled preds/targets).
Validation metric = mean over the 183 tracks of per-track Pearson r pooled over all bins of the split, on the shared 896 x 128 bp
grid (crop 256 bins/side of the central 4096, mean-pool x4; same sums as analysis/priorityD/headlib.predict_tiles + estats.pearson_from_sums).
Reported also: mean r over the 152 non-identical tracks (exposure_audit, Borzoi_class != identical_experiment).  Pearson r only here.

Usage: train_ft.py --run-id RUN --mode partial --unfreeze blocks:2 ...   (resumes automatically from results_ft/runs/RUN/ckpts)
Exit codes: 0 finished, 3 launch refused by deadline policy, 4 storage guard, 5 OOM, 6 non-finite loss, 7 stopped by signal/request.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import csv
import fcntl
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "analysis/exposure_stratified"))
import ckpt as C
import data_ft as D
import model_ft as M
import policy as P
from estats import pearson_from_sums

REGISTRY = ROOT / "research_management/finetuning/FT_REGISTRY.csv"
REG_HEADER = ["exp_id", "method", "config", "train", "val", "seed", "start", "end", "runtime_s", "gpu_mem_gb", "checkpoint",
              "val_r_all183", "val_r_152", "test_r_all183", "test_r_152", "status", "notes"]
CROP, POOL, G_BINS = 256, 4, 896


def parse():
    a = argparse.ArgumentParser()
    a.add_argument("--run-id", required=True)
    a.add_argument("--out-root", default=str(C.RUNS_ROOT))
    a.add_argument("--mode", choices=["frozen", "partial", "lora", "full"], required=True)
    a.add_argument("--replicate", type=int, choices=[0, 1, 2, 3], default=0, help="Borzoi replicate (johahi/borzoi-replicate-N)")
    a.add_argument("--unfreeze", default="blocks:2")
    a.add_argument("--lora-targets", default="attn")
    a.add_argument("--lora-r", type=int, default=8)
    a.add_argument("--lora-alpha", type=float, default=16.0)
    a.add_argument("--lora-blocks", type=int, default=8)
    a.add_argument("--head", choices=["mlp", "linear"], default="mlp")
    a.add_argument("--head-hidden", type=int, default=1024)
    a.add_argument("--head-dropout", type=float, default=0.2)
    a.add_argument("--loss", choices=["poisson", "mse128"], default="poisson")
    a.add_argument("--lr-head", type=float, default=1e-4)
    a.add_argument("--lr-trunk", type=float, default=1e-5)
    a.add_argument("--lr-lora", type=float, default=1e-4)
    a.add_argument("--wd", type=float, default=1e-4)
    a.add_argument("--trunk-wd", type=float, default=0.0)
    a.add_argument("--clip", type=float, default=1.0)
    a.add_argument("--sched", choices=["constant", "cosine"], default="cosine")
    a.add_argument("--warmup-steps", type=int, default=200)
    a.add_argument("--batch-size", type=int, default=1)
    a.add_argument("--accum", type=int, default=1)
    a.add_argument("--epochs", type=int, default=4)
    a.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps (0 = off)")
    a.add_argument("--patience", type=int, default=0, help="early stop on full-val mean r (epochs; 0 = off)")
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--rc-prob", type=float, default=0.5)
    a.add_argument("--max-shift-bins", type=int, default=4)
    a.add_argument("--grad-ckpt", choices=["none", "transformer", "all"], default="transformer")
    a.add_argument("--trunk-dropout", action="store_true")
    a.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    a.add_argument("--deterministic", action="store_true")
    a.add_argument("--num-workers", type=int, default=6)
    a.add_argument("--ckpt-every-min", type=float, default=30.0)
    a.add_argument("--ckpt-every-steps", type=int, default=0)
    a.add_argument("--val-every-steps", type=int, default=0, help="subset validation every N steps (log only)")
    a.add_argument("--val-subset-n", type=int, default=150)
    a.add_argument("--log-every", type=int, default=10)
    a.add_argument("--train-limit", type=int, default=0, help="debug: first N train tiles")
    a.add_argument("--val-limit", type=int, default=0, help="debug: evenly spaced N val tiles for 'full' validation")
    a.add_argument("--est-hours", type=float, default=0.0, help="estimated duration, checked against the deadline policy")
    a.add_argument("--mem-cap-gb", type=float, default=0.0)
    a.add_argument("--profile-steps", type=int, default=0, help="profiling mode: 2 warm-up + N timed steps, ckpt write, val timing")
    a.add_argument("--resume", choices=["auto", "never"], default="auto")
    a.add_argument("--stop-after-steps", type=int, default=0, help="test hook: exit cleanly (with checkpoint) after N steps this process")
    a.add_argument("--ckpt-max-gb", type=float, default=0.0)
    return a.parse_args()


# ----------------------------------------------------------------------------------------------------------------- helpers
def registry_append(row: dict):
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        w = csv.DictWriter(f, fieldnames=REG_HEADER)
        if f.tell() == 0:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in REG_HEADER})
        fcntl.flock(f, fcntl.LOCK_UN)


def loss_fn(pred, tgt, kind):
    if kind == "poisson":
        return F.poisson_nll_loss(pred, tgt, log_input=False)
    p = F.avg_pool1d(pred.permute(0, 2, 1), 4).permute(0, 2, 1)
    t = F.avg_pool1d(tgt.permute(0, 2, 1), 4).permute(0, 2, 1)
    return F.mse_loss(p, t)


def make_sched(opt, total_steps, warmup, kind):
    def f(s):
        w = min(1.0, (s + 1) / max(1, warmup)) if warmup else 1.0
        if kind == "constant":
            return w
        prog = min(1.0, s / max(1, total_steps))
        return w * (0.5 * (1 + math.cos(math.pi * prog)) * 0.99 + 0.01)
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def mask152():
    import pandas as pd
    e = pd.read_csv(ROOT / "exposure_audit/track_exposure.tsv", sep="\t")
    assert (e.track_idx.values == np.arange(183)).all()
    return (e.Borzoi_class != "identical_experiment").values


@torch.no_grad()
def evaluate(model, tiles, gids, args, abort=lambda: False, tag="val"):
    """Mean per-track Pearson r on shared128 (fwd, unshifted, no RC). Returns dict or None if aborted."""
    model.eval()
    dev = "cuda"
    dl = DataLoader(D.TileJobs(tiles, [(int(g), False, 0) for g in gids]), batch_size=1, num_workers=min(4, args.num_workers))
    S = {k: torch.zeros(183, dtype=torch.float64, device=dev) for k in ["sx", "sy", "sxy", "sx2", "sy2", "sse"]}
    n_tiles, losses, t0 = 0, [], time.time()
    for i, b in enumerate(dl):
        if i % 20 == 0 and abort():
            return None
        pred = model(b["codes"].to(dev), autocast=args.precision == "bf16")
        tgt = b["tgt"].to(dev)
        losses.append(loss_fn(pred, tgt, args.loss).item())
        P_ = pred[0].double()[CROP:CROP + G_BINS * POOL].reshape(G_BINS, POOL, -1).mean(1)
        Y_ = tgt[0].double()[CROP:CROP + G_BINS * POOL].reshape(G_BINS, POOL, -1).mean(1)
        S["sx"] += P_.sum(0); S["sy"] += Y_.sum(0); S["sxy"] += (P_ * Y_).sum(0)
        S["sx2"] += (P_ * P_).sum(0); S["sy2"] += (Y_ * Y_).sum(0); S["sse"] += ((P_ - Y_) ** 2).sum(0)
        n_tiles += 1
    n = np.full(183, float(n_tiles * G_BINS))
    g = {k: v.cpu().numpy() for k, v in S.items()}
    r = pearson_from_sums(n, g["sx"], g["sy"], g["sxy"], g["sx2"], g["sy2"])
    m152 = mask152()
    return dict(tag=tag, n_tiles=n_tiles, loss=float(np.mean(losses)), mean_r_all183=float(np.nanmean(r)),
                mean_r_152=float(np.nanmean(r[m152])), n_nan_tracks=int(np.isnan(r).sum()), seconds=time.time() - t0,
                per_track_r=[None if np.isnan(x) else float(x) for x in r])


def build(args, tiles, repo_id):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg = vars(args)
    model = M.FTModel(cfg, repo_id=repo_id).cuda()
    groups = model.param_groups(args.lr_head, args.lr_trunk, args.lr_lora, args.wd, args.trunk_wd)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    return model, opt


def log_line(run_dir, msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)


# ----------------------------------------------------------------------------------------------------------------- main
def main():
    args = parse()
    repo_id = f"johahi/borzoi-replicate-{args.replicate}"
    cfg = vars(args)
    run_dir = Path(args.out_root) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log = lambda m: log_line(run_dir, m)
    t_start = time.time()
    if args.deterministic or args.precision == "fp32":
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    if args.precision == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    if args.mem_cap_gb > 0:
        torch.cuda.set_per_process_memory_fraction(args.mem_cap_gb / (torch.cuda.get_device_properties(0).total_memory / 1024 ** 3))

    # ---- data tables
    tiles = D.load_tiles()
    train_gids = tiles[tiles.split == "train"].gid.to_numpy()
    val_gids = tiles[tiles.split == "val"].gid.to_numpy()
    if args.train_limit:
        train_gids = train_gids[:args.train_limit]
    if args.val_limit:
        val_gids = val_gids[np.linspace(0, len(val_gids) - 1, args.val_limit).astype(int)]
    val_sub = val_gids[np.linspace(0, len(val_gids) - 1, min(args.val_subset_n, len(val_gids))).astype(int)]
    assert set(tiles[tiles.split == "test"].chrom) == {"chr9", "chr22"} and not set(tiles.loc[tiles.gid.isin(train_gids), "chrom"]) & {"chr9", "chr22", "chr1", "chr8"}

    # ---- model, optimizer
    model, opt = build(args, tiles, repo_id)
    steps_per_epoch = math.ceil(len(train_gids) / (args.batch_size * args.accum))
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    sched = make_sched(opt, total_steps, args.warmup_steps, args.sched)
    mgr = C.CheckpointManager(run_dir, cfg, model.trunk_arch, M.frozen_manifest(repo_id), D.tiles_hash(), keep=2,
                              est_bytes=(model.n_trainable() * 12 + 1e8), log=log)
    log(f"run {args.run_id}: mode={args.mode} trainable={model.n_trainable():,} first_train_stage={model.first_train_stage} "
        f"lora_modules={len(model.lora_modules)} steps/epoch={steps_per_epoch} total_steps={total_steps} cfg_hash={mgr.chash}")

    # ---- resume
    st = dict(epoch=0, global_step=0, epoch_pos=0, epoch_order=None, best_val=-1.0, no_improve=0, pending_val=False, history=[],
              train_seconds=0.0, started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    resumed = False
    if args.resume == "auto" and mgr.list_ckpts():
        f, payload = mgr.find_latest_valid()
        if payload is None:
            C.write_status(run_dir, "failed", reason="checkpoints exist but none valid/compatible")
            log("FATAL: checkpoints exist but none valid; refusing to restart from scratch silently"); sys.exit(8)
        st = mgr.restore(payload, model, opt, sched, None)
        resumed = True
        (run_dir / "resume_verify.json").write_text(json.dumps(dict(
            ckpt=f.name, global_step=st["global_step"], epoch=st["epoch"], epoch_pos=st["epoch_pos"],
            opt_state_sha=_opt_hash(opt))))
        log(f"RESUMED from {f.name}: epoch {st['epoch']} pos {st['epoch_pos']} global_step {st['global_step']}")
    ok, why = P.launch_allowed(args.est_hours, resume=resumed)
    if not ok:
        C.write_status(run_dir, "refused", reason=why); log(f"launch refused: {why}"); sys.exit(3)
    (run_dir / "run_cmd.json").write_text(json.dumps(dict(argv=sys.argv[1:], python=sys.executable)))

    if args.profile_steps:
        C.write_status(run_dir, "running", profile=True)
        return profile(args, model, opt, sched, mgr, tiles, train_gids, val_gids, run_dir, log, t_start)

    trig = C.CheckpointTriggers(run_dir, args.ckpt_every_min, args.ckpt_every_steps)
    C.write_status(run_dir, "running", resumed=resumed, global_step=st["global_step"])
    t_proc0, steps_this_proc, ema, last_prog = time.time(), 0, None, 0.0
    exit_code, final_state = 0, "finished"

    def progress(extra=None):
        rem_steps = max(0, total_steps - st["global_step"])
        d = dict(run_id=args.run_id, pid=os.getpid(), time=time.time(), epoch=st["epoch"], global_step=st["global_step"],
                 epoch_pos=st["epoch_pos"], steps_per_epoch=steps_per_epoch, total_steps=total_steps,
                 s_per_step_ema=ema, est_remaining_s=(ema * rem_steps if ema else None), last_ckpt_time=mgr.last_save_time,
                 last_ckpt_step=mgr.last_save_step, ckpt_every_s=args.ckpt_every_min * 60, best_val=st["best_val"],
                 gpu_mem_gb=torch.cuda.max_memory_allocated() / 1024 ** 3)
        d.update(extra or {})
        C.atomic_write_text(run_dir / "progress.json", json.dumps(d))

    def save(reason):
        st["train_seconds"] = st.get("train_seconds", 0.0)
        mgr.save(model, opt, sched, None, st, reason=reason)
        trig.done()
        progress()

    metrics_f = open(run_dir / "metrics.csv", "a", newline="")
    mw = csv.writer(metrics_f)
    if metrics_f.tell() == 0:
        mw.writerow(["global_step", "epoch", "epoch_pos", "loss", "grad_norm", "lr_head", "lr_trunk", "sec_per_step", "time"])

    try:
        stop = False
        while st["epoch"] < args.epochs and not stop:
            jobs = D.epoch_jobs(train_gids, st["epoch"], args.seed, args.rc_prob, args.max_shift_bins)
            gl = [j[0] for j in jobs]
            if st["epoch_order"] is not None:
                assert st["epoch_order"] == gl, "saved sampler order differs from regenerated order"
            st["epoch_order"] = gl
            if not st["pending_val"]:
                pos0 = st["epoch_pos"]
                dl = DataLoader(D.TileJobs(tiles, jobs[pos0:]), batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, prefetch_factor=4 if args.num_workers else None)
                model.train()
                acc, run_loss, bad = 0, 0.0, 0
                n_jobs = len(jobs)
                t_step = time.time()
                for bi, b in enumerate(dl):
                    pred = model(b["codes"].cuda(non_blocking=True), autocast=args.precision == "bf16")
                    loss = loss_fn(pred, b["tgt"].cuda(non_blocking=True), args.loss)
                    if not torch.isfinite(loss):
                        bad += 1; opt.zero_grad(set_to_none=True); acc, run_loss = 0, 0.0
                        log(f"non-finite loss at step {st['global_step']} ({bad} consecutive)")
                        if bad >= 20:
                            raise FloatingPointError("20 consecutive non-finite losses")
                        st["epoch_pos"] += b["codes"].shape[0]
                        continue
                    bad = 0
                    (loss / args.accum).backward()
                    acc += 1; run_loss += loss.item()
                    st["epoch_pos"] += b["codes"].shape[0]
                    last = st["epoch_pos"] >= n_jobs
                    if acc < args.accum and not last:
                        continue
                    gn = torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], args.clip)
                    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                    st["global_step"] += 1; steps_this_proc += 1
                    dt = time.time() - t_step; t_step = time.time()
                    ema = dt if ema is None else 0.95 * ema + 0.05 * dt
                    mw.writerow([st["global_step"], st["epoch"], st["epoch_pos"], f"{run_loss / acc:.6f}", f"{float(gn):.4f}",
                                 f"{opt.param_groups[0]['lr']:.3e}", f"{opt.param_groups[-1]['lr']:.3e}", f"{dt:.3f}", f"{time.time():.1f}"])
                    metrics_f.flush()
                    if st["global_step"] % args.log_every == 0:
                        log(f"ep {st['epoch']} step {st['global_step']}/{total_steps} pos {st['epoch_pos']}/{n_jobs} loss {run_loss / acc:.4f} "
                            f"gn {float(gn):.2f} {ema:.2f}s/step mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")
                    acc, run_loss = 0, 0.0
                    if time.time() - last_prog > 30:
                        progress(); last_prog = time.time()
                    if args.val_every_steps and st["global_step"] % args.val_every_steps == 0:
                        r = evaluate(model, tiles, val_sub, args, abort=trig.stop_requested, tag="val_subset")
                        model.train()
                        if r:
                            r.pop("per_track_r"); r["global_step"] = st["global_step"]; st["history"].append(r)
                            log(f"val-subset({r['n_tiles']}) step {st['global_step']}: mean r183={r['mean_r_all183']:.4f} r152={r['mean_r_152']:.4f}")
                    if os.environ.get("FT_TEST_SLEEP_PER_STEP"):          # test hook only
                        time.sleep(float(os.environ["FT_TEST_SLEEP_PER_STEP"]))
                    do_ck, do_stop, why = trig.poll(st["global_step"])
                    reached_max = bool(args.max_steps and st["global_step"] >= args.max_steps)
                    reached_proc = bool(args.stop_after_steps and steps_this_proc >= args.stop_after_steps)
                    if do_ck or do_stop or reached_max or reached_proc:
                        save(why or ("max-steps" if reached_max else "stop-after-steps"))
                    if do_stop:
                        exit_code, final_state, stop = 7, "stopped", True
                    elif reached_max or reached_proc:
                        exit_code, final_state, stop = 0, ("finished" if reached_max else "stopped_test_hook"), True
                    if stop:
                        break
                if stop:
                    break
                st["pending_val"] = True
                st["epoch_pos"] = n_jobs
            # ---- epoch end: full validation (resumable: pending_val)
            r = evaluate(model, tiles, val_gids, args, abort=trig.stop_requested, tag="val_full")
            if r is None:
                save(trig.reason or "stop-during-val"); exit_code, final_state, stop = 7, "stopped", True
                break
            r["epoch"] = st["epoch"] + 1; r["global_step"] = st["global_step"]
            st["history"].append({k: v for k, v in r.items() if k != "per_track_r"})
            (run_dir / f"val_epoch{st['epoch'] + 1}_per_track_r.json").write_text(json.dumps(r["per_track_r"]))
            log(f"EPOCH {st['epoch'] + 1} full val ({r['n_tiles']} tiles, {r['seconds']:.0f}s): loss {r['loss']:.4f} "
                f"mean r183 {r['mean_r_all183']:.4f} r152 {r['mean_r_152']:.4f}")
            if r["mean_r_all183"] > st["best_val"]:
                st["best_val"], st["no_improve"] = r["mean_r_all183"], 0
                st["best_val_152"] = r["mean_r_152"]
                mgr.save_best(model, r["mean_r_all183"], dict(epoch=st["epoch"] + 1, global_step=st["global_step"]))
            else:
                st["no_improve"] += 1
            st["epoch"] += 1; st["epoch_pos"] = 0; st["epoch_order"] = None; st["pending_val"] = False
            save("epoch-end")
            if args.patience and st["no_improve"] >= args.patience:
                log("early stopping"); break
            if args.max_steps and st["global_step"] >= args.max_steps:
                break
            do_ck, do_stop, why = trig.poll(st["global_step"])
            if do_stop:
                exit_code, final_state, stop = 7, "stopped", True
        if final_state == "finished" and (mgr.last_save_step != st["global_step"]):
            save("final")
    except C.StorageGuardError as e:
        exit_code, final_state = 4, "stopped_storage_guard"
        log(f"stopping cleanly: {e}")
    except torch.OutOfMemoryError as e:
        exit_code, final_state = 5, "failed_oom"
        log(f"OOM: {e}")
    except FloatingPointError as e:
        exit_code, final_state = 6, "failed_nonfinite"
        log(str(e))
    except BaseException as e:  # noqa
        exit_code, final_state = 1, "failed"
        log("EXCEPTION " + traceback.format_exc())
    finally:
        metrics_f.close()
    if final_state != "stopped_storage_guard":
        C.write_status(run_dir, final_state, global_step=st["global_step"], epoch=st["epoch"], best_val=st["best_val"],
                       exit_code=exit_code)
    st_now = time.strftime("%Y-%m-%dT%H:%M:%S")
    if final_state in ("finished", "failed", "failed_oom", "failed_nonfinite", "stopped_storage_guard"):
        registry_append(dict(exp_id=args.run_id, method=f"{args.mode}:{args.unfreeze if args.mode == 'partial' else ''}",
                             config=json.dumps({k: v for k, v in cfg.items() if k not in C.VOLATILE_KEYS}), train=len(train_gids),
                             val=len(val_gids), seed=args.seed, start=st["started"], end=st_now,
                             runtime_s=int(time.time() - t_start), gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 1024 ** 3, 1),
                             checkpoint=str(run_dir / "best_model.pt"), val_r_all183=round(st["best_val"], 4) if st["best_val"] > 0 else "",
                             val_r_152=round(st.get("best_val_152", float("nan")), 4) if "best_val_152" in st else "", status=final_state,
                             notes=f"steps={st['global_step']} epochs_done={st['epoch']}"))
    log(f"exit state={final_state} code={exit_code}")
    sys.exit(exit_code)


def _opt_hash(opt):
    import hashlib
    h = hashlib.sha256()
    sd = opt.state_dict()
    for k in sorted(sd["state"].keys()):
        for kk in sorted(sd["state"][k].keys()):
            v = sd["state"][k][kk]
            h.update(v.detach().cpu().numpy().tobytes() if torch.is_tensor(v) else str(v).encode())
    h.update(json.dumps(sd["param_groups"], default=str).encode())
    return h.hexdigest()[:16]


def profile(args, model, opt, sched, mgr, tiles, train_gids, val_gids, run_dir, log, t_start):
    """2 warm-up + N timed steps on the first training tiles; one timed checkpoint write; validation timing on args.val_limit tiles."""
    res = dict(run_id=args.run_id, mode=args.mode, trainable_params=model.n_trainable(), first_train_stage=model.first_train_stage,
               grad_ckpt=args.grad_ckpt, precision=args.precision, fits=True, config={k: v for k, v in vars(args).items()})
    jobs = D.epoch_jobs(train_gids, 0, args.seed, args.rc_prob, args.max_shift_bins)[:args.profile_steps + 2]
    dl = DataLoader(D.TileJobs(tiles, jobs), batch_size=1, num_workers=args.num_workers, pin_memory=True)
    times, losses = [], []
    torch.cuda.reset_peak_memory_stats()
    status = "profile"
    try:
        model.train()
        for i, b in enumerate(dl):
            torch.cuda.synchronize(); t0 = time.time()
            pred = model(b["codes"].cuda(), autocast=args.precision == "bf16")
            loss = loss_fn(pred, b["tgt"].cuda(), args.loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], args.clip)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); dt = time.time() - t0
            losses.append(loss.item())
            if i >= 2:
                times.append(dt)
            log(f"profile step {i}: {dt:.2f}s loss {loss.item():.4f} peak {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")
        res.update(peak_alloc_gb=torch.cuda.max_memory_allocated() / 1024 ** 3, peak_reserved_gb=torch.cuda.max_memory_reserved() / 1024 ** 3,
                   s_per_step_mean=float(np.mean(times)), s_per_step_median=float(np.median(times)), s_per_step_all=times, losses=losses)
        st = dict(epoch=0, global_step=len(jobs), epoch_pos=0, epoch_order=None, best_val=-1.0, no_improve=0, pending_val=False, history=[])
        t0 = time.time()
        p = mgr.save(model, opt, sched, None, st, reason="profile")
        res.update(ckpt_write_s=time.time() - t0, ckpt_bytes=p.stat().st_size)
        res["epoch_est_h_4659_tiles"] = res["s_per_step_mean"] * 4659 / (args.batch_size * args.accum) * args.accum / 3600  # s/step is per micro-batch
        r = evaluate(model, tiles, val_gids[np.linspace(0, len(val_gids) - 1, min(20, len(val_gids))).astype(int)], args, tag="val_profile")
        r.pop("per_track_r")
        res.update(val_20tiles_s=r["seconds"], val_750_est_min=r["seconds"] / r["n_tiles"] * 750 / 60,
                   val_peak_alloc_gb=torch.cuda.max_memory_allocated() / 1024 ** 3)
    except torch.OutOfMemoryError as e:
        res.update(fits=False, error="OOM: " + str(e)[:300], peak_alloc_gb=torch.cuda.max_memory_allocated() / 1024 ** 3)
        status = "oom"
        log("OOM " + str(e)[:200])
    except BaseException:
        res.update(fits=False, error=traceback.format_exc()[-800:])
        status = "failed"
        log(traceback.format_exc())
    (run_dir / "profile.json").write_text(json.dumps(res, indent=1, default=str))
    registry_append(dict(exp_id=args.run_id, method=f"profile:{args.mode}", config=json.dumps(
        {k: v for k, v in vars(args).items() if k in ("mode", "unfreeze", "lora_targets", "lora_r", "lora_blocks", "head", "loss", "grad_ckpt", "precision")}),
        train=len(train_gids), val=len(val_gids), seed=args.seed, start="", end=time.strftime("%Y-%m-%dT%H:%M:%S"),
        runtime_s=int(time.time() - t_start), gpu_mem_gb=round(res.get("peak_alloc_gb", 0), 1), checkpoint="", status=status,
        notes=f"s/step={res.get('s_per_step_mean', float('nan')):.2f}; trainable={res['trainable_params']}; ckpt_MB={res.get('ckpt_bytes', 0) / 1e6:.0f}; "
              f"profile-only (no val r)"))
    C.write_status(run_dir, "profile_done" if status == "profile" else status)
    sys.exit(0 if status == "profile" else 5)


if __name__ == "__main__":
    main()
