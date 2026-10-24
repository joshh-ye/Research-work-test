"""Checkpoint / resume library for Borzoi fine-tuning (analysis/ft).

Design
- Atomic writes: tmp file in the SAME directory -> flush + fsync -> os.replace -> fsync(dir).
- Resumable checkpoints `ckpt_step{N:08d}.pt` (keep the 2 most recent) + a separate `best_model.pt` (best FULL validation
  mean Pearson r).  Both store ONLY trainable parameters (head + whatever trunk parameters / LoRA matrices are trainable) plus
  a manifest of the frozen source (HF repo id + snapshot revision hash + file size); the frozen trunk is re-loaded from the
  local HF cache on resume.  Resumable checkpoints additionally hold optimizer, scheduler, scaler (None under bf16),
  epoch / global step / position inside the epoch (+ the epoch's sample order), RNG states (torch CPU, CUDA, numpy, python),
  config, arch config, best val metric, split hash (sha256 of tiles.csv), git commit, package versions.
- find_latest_valid(): newest-first, fully loads each file, checks format/keys and config-compatibility hash.
- Storage guard (home quota): usage is read from `accre_storage` (the cluster quota tool; ~0.1 s; the `Home (user)` row =
  Usage / soft Quota).  `du` is NOT used (225k files).  Fallback if the tool is missing/unparseable: statvfs free space of the
  target filesystem (panfs reports the whole file system, so this only guards against a truly full disk) - a WARNING is logged.
  Rules: refuse a write if home usage > 175 GB, or if soft_quota - usage - (estimated new checkpoint) < 5 GB.  Estimated
  new size = size of the newest existing checkpoint (or `est_bytes` for the first one).  Peak usage is counted conservatively:
  the oldest checkpoint is deleted only AFTER the new one is in place.  On refusal the previous checkpoints are kept untouched,
  `STATUS.json` gets state=stopped_storage_guard and `StorageGuardError` is raised (train_ft exits cleanly with code 4).
  Test hook: env FT_STORAGE_FAKE_USAGE_GB overrides the measured usage.
- Wall-clock / signal triggers live in `CheckpointTriggers` (30 min default, epoch end handled by the caller, SIGTERM/SIGUSR1,
  and file requests `REQUEST_CKPT` / `REQUEST_STOP` in the run dir written by the supervisor).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

FORMAT_VERSION = 2
ROOT = Path("/home/yejz1/experimental_research")
RUNS_ROOT = ROOT / "results_ft" / "runs"
HOME_LIMIT_GB = 175.0
MIN_FREE_AFTER_GB = 5.0

# config keys that do NOT define model/optimizer structure (may differ between a run and its resume)
VOLATILE_KEYS = {"epochs", "max_steps", "ckpt_every_min", "ckpt_every_steps", "val_every_steps", "val_subset_n", "log_every",
                 "num_workers", "run_id", "out_root", "resume", "profile_steps", "est_hours", "train_limit", "val_limit",
                 "patience", "deterministic", "fake_crash_after", "stop_after_steps", "ckpt_max_gb"}


class StorageGuardError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------------------------- utils
# Options added after runs had started: omitted from the hash at their default, so those runs stay resumable.
COMPAT_DEFAULTS = {"replicate": 0}


def config_hash(cfg: dict) -> str:
    c = {k: v for k, v in sorted(cfg.items())
         if k not in VOLATILE_KEYS and not (k in COMPAT_DEFAULTS and v == COMPAT_DEFAULTS[k])}
    return hashlib.sha256(json.dumps(c, sort_keys=True, default=str).encode()).hexdigest()[:16]


def file_sha256(path, n=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read() if n is None else f.read(n))
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, timeout=10).strip()
    except Exception:
        return "unknown"


def package_versions() -> dict:
    v = {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda}
    for m in ["borzoi_pytorch", "transformers", "pyfaidx", "pyBigWig"]:
        try:
            from importlib import metadata
            v[m] = metadata.version(m.replace("_", "-") if m == "borzoi_pytorch" else m)
        except Exception:
            v[m] = "?"
    return v


def atomic_torch_save(obj, path: Path):
    """tmp file in the same dir -> fsync -> os.replace -> fsync dir. Returns size in bytes."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        size = tmp.stat().st_size
        os.replace(tmp, path)
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
        return size
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_text(path: Path, text: str):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------------------------------- storage
_UNITS = {"M": 1 / 1024, "G": 1.0, "T": 1024.0, "P": 1024.0 ** 2, "K": 1 / 1024 ** 2}


def _to_gb(s: str) -> float:
    m = re.match(r"([0-9.]+)\s*([KMGTP])", s.strip())
    return float(m.group(1)) * _UNITS[m.group(2)]


_storage_cache = {"t": 0.0, "v": None}


def home_usage_gb(max_age_s: float = 20.0):
    """(usage_GB, soft_quota_GB, source). Source = 'FAKE' | 'accre_storage' | 'statvfs-fallback'."""
    fake = os.environ.get("FT_STORAGE_FAKE_USAGE_GB")
    if fake:
        return float(fake), 186.3, "FAKE"
    now = time.time()
    if _storage_cache["v"] is not None and now - _storage_cache["t"] < max_age_s:
        return _storage_cache["v"]
    try:
        out = subprocess.check_output(["accre_storage"], text=True, timeout=30, stderr=subprocess.STDOUT)
        for line in out.splitlines():
            if line.strip().startswith("Home"):
                toks = line.split()
                # "Home (user): 118.4G 186.3G 232.8G ..."
                vals = [t for t in toks if re.match(r"^[0-9.]+[KMGTP]$", t)]
                v = (_to_gb(vals[0]), _to_gb(vals[1]), "accre_storage")
                _storage_cache.update(t=now, v=v)
                return v
        raise ValueError("Home row not found")
    except Exception as e:  # pragma: no cover
        print(f"[ckpt] WARNING accre_storage unavailable ({e}); falling back to statvfs", file=sys.stderr)
        st = os.statvfs(ROOT)
        free = st.f_bavail * st.f_frsize / 1024 ** 3
        # statvfs on panfs reports the whole file system: only a truly full disk (<10 GB free) trips the guard
        return (176.0 if free < 10 else 0.0), 186.3, "statvfs-fallback"


def storage_check(est_bytes: float, log=print):
    """Raise StorageGuardError if a write of est_bytes violates the guard."""
    use, quota, src = home_usage_gb()
    est = est_bytes / 1024 ** 3
    if use > HOME_LIMIT_GB:
        raise StorageGuardError(f"home usage {use:.1f} GB > {HOME_LIMIT_GB} GB limit ({src})")
    if quota - use - est < MIN_FREE_AFTER_GB:
        raise StorageGuardError(f"write of {est:.2f} GB would leave {quota - use - est:.1f} GB < {MIN_FREE_AFTER_GB} GB "
                                f"(usage {use:.1f}/{quota:.1f} GB, {src})")
    return use, quota, src


# ----------------------------------------------------------------------------------------------------------------- status
def write_status(run_dir: Path, state: str, **kw):
    d = dict(state=state, time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), pid=os.getpid(), **kw)
    atomic_write_text(Path(run_dir) / "STATUS.json", json.dumps(d, indent=1, default=str))


def read_status(run_dir: Path):
    try:
        return json.loads((Path(run_dir) / "STATUS.json").read_text())
    except Exception:
        return None


# ----------------------------------------------------------------------------------------------------------------- manager
def rng_states() -> dict:
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                numpy=np.random.get_state(), python=random.getstate())


def set_rng_states(s: dict):
    torch.set_rng_state(s["torch"].cpu())
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in s["cuda"]])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])


def trainable_state(model: torch.nn.Module) -> dict:
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in names}


class CheckpointManager:
    def __init__(self, run_dir, config: dict, arch_cfg: dict, frozen_manifest: dict, split_hash: str, keep: int = 2,
                 est_bytes: float = 2e9, log=print):
        self.run_dir = Path(run_dir)
        self.ck_dir = self.run_dir / "ckpts"
        self.ck_dir.mkdir(parents=True, exist_ok=True)
        self.config, self.arch_cfg, self.frozen_manifest, self.split_hash = config, arch_cfg, frozen_manifest, split_hash
        self.chash = config_hash(config)
        self.keep, self.est_bytes, self.log = keep, est_bytes, log
        self.last_save_time = time.time()
        self.last_save_step = -1

    # ---- listing / validation
    def list_ckpts(self):
        fs = sorted(self.ck_dir.glob("ckpt_step*.pt"))
        return fs

    def verify(self, path):
        """Return (ok, payload_or_reason). Full torch.load (zip central directory => detects truncation)."""
        try:
            p = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            return False, f"unloadable: {type(e).__name__}: {e}"
        need = ["format", "meta", "trainable", "optimizer", "scheduler", "state", "rng", "config", "config_hash"]
        miss = [k for k in need if k not in p]
        if miss:
            return False, f"missing keys {miss}"
        if p["format"] != FORMAT_VERSION:
            return False, f"format {p['format']} != {FORMAT_VERSION}"
        if p["config_hash"] != self.chash:
            return False, f"config hash {p['config_hash']} != {self.chash} (incompatible config)"
        if p["meta"]["split_hash"] != self.split_hash:
            return False, "split hash mismatch (tiles.csv changed)"
        return True, p

    def find_latest_valid(self):
        for f in reversed(self.list_ckpts()):
            ok, p = self.verify(f)
            if ok:
                return f, p
            self.log(f"[ckpt] skipping {f.name}: {p}")
        return None, None

    # ---- saving
    def _guard(self):
        ck = self.list_ckpts()
        est = ck[-1].stat().st_size if ck else self.est_bytes
        try:
            storage_check(est + 0.0, self.log)
        except StorageGuardError as e:
            write_status(self.run_dir, "stopped_storage_guard", reason=str(e), kept=[c.name for c in ck])
            self.log(f"[ckpt] STORAGE GUARD refused write: {e}; previous checkpoints kept: {[c.name for c in ck]}")
            raise

    def save(self, model, optimizer, scheduler, scaler, state: dict, extra_meta: dict | None = None, reason: str = "") -> Path:
        """state: epoch, global_step, epoch_pos, epoch_order (list), best_val, no_improve, pending_val, history (any json)."""
        self._guard()
        t0 = time.time()
        payload = dict(
            format=FORMAT_VERSION,
            meta=dict(saved_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), reason=reason, git_commit=git_commit(),
                      versions=package_versions(), split_hash=self.split_hash, frozen_source=self.frozen_manifest,
                      arch_cfg=self.arch_cfg, **(extra_meta or {})),
            trainable=trainable_state(model),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict() if scheduler is not None else None,
            scaler=scaler.state_dict() if scaler is not None else None,
            state=state,
            rng=rng_states(),
            config=self.config, config_hash=self.chash,
        )
        path = self.ck_dir / f"ckpt_step{state['global_step']:08d}.pt"
        size = atomic_torch_save(payload, path)
        # prune only after the new file is safely in place
        for old in self.list_ckpts()[:-self.keep]:
            old.unlink()
        self.last_save_time, self.last_save_step = time.time(), state["global_step"]
        self.log(f"[ckpt] saved {path.name} ({size / 1e6:.0f} MB, {time.time() - t0:.1f}s, reason={reason})")
        return path

    def save_best(self, model, val_metric: float, state: dict):
        self._guard()
        payload = dict(format=FORMAT_VERSION, kind="best_model",
                       meta=dict(saved_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), git_commit=git_commit(), versions=package_versions(),
                                 split_hash=self.split_hash, frozen_source=self.frozen_manifest, arch_cfg=self.arch_cfg,
                                 val_mean_pearson_r_full=val_metric, epoch=state["epoch"], global_step=state["global_step"]),
                       trainable=trainable_state(model), config=self.config, config_hash=self.chash)
        size = atomic_torch_save(payload, self.run_dir / "best_model.pt")
        self.log(f"[ckpt] best_model.pt updated (val r={val_metric:.4f}, {size / 1e6:.0f} MB)")

    # ---- restoring
    def restore(self, payload, model, optimizer, scheduler, scaler):
        missing, unexpected = model.load_state_dict(payload["trainable"], strict=False)
        assert not unexpected, f"unexpected keys {unexpected[:3]}"
        optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload["scheduler"] is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None and payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        set_rng_states(payload["rng"])
        return payload["state"]


# ----------------------------------------------------------------------------------------------------------------- triggers
class CheckpointTriggers:
    """SIGTERM -> checkpoint then stop; SIGUSR1 -> checkpoint and continue; files REQUEST_CKPT / REQUEST_STOP in run dir
    (supervisor).  Poll `poll()` at optimizer-step boundaries."""

    def __init__(self, run_dir, every_min: float = 30.0, every_steps: int = 0):
        self.run_dir, self.every_s, self.every_steps = Path(run_dir), every_min * 60.0, every_steps
        self.want_ckpt = False
        self.want_stop = False
        self.reason = ""
        self.last = time.time()
        signal.signal(signal.SIGTERM, self._term)
        signal.signal(signal.SIGUSR1, self._usr1)

    def _term(self, *_):
        self.want_ckpt = self.want_stop = True
        self.reason = "SIGTERM"

    def _usr1(self, *_):
        self.want_ckpt = True
        self.reason = self.reason or "SIGUSR1"

    def poll(self, global_step: int):
        """-> (do_ckpt, do_stop, reason)"""
        rc, rs = self.run_dir / "REQUEST_CKPT", self.run_dir / "REQUEST_STOP"
        if rc.exists():
            self.want_ckpt, self.reason = True, self.reason or "supervisor-request"
            rc.unlink(missing_ok=True)
        if rs.exists():
            self.want_ckpt = self.want_stop = True
            self.reason = "supervisor-stop"
        if not self.want_ckpt and time.time() - self.last >= self.every_s:
            self.want_ckpt, self.reason = True, "wall-clock"
        if not self.want_ckpt and self.every_steps and global_step > 0 and global_step % self.every_steps == 0:
            self.want_ckpt, self.reason = True, "every-steps"
        r = (self.want_ckpt, self.want_stop, self.reason)
        return r

    def stop_requested(self) -> bool:
        return self.want_stop or (self.run_dir / "REQUEST_STOP").exists()

    def done(self):
        self.want_ckpt, self.reason, self.last = False, "", time.time()
