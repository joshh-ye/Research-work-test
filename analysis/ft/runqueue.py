"""Run queue for the fine-tuning supervisor.

research_management/finetuning/QUEUE.json = ordered JSON list of run entries (the coordinator edits/enqueues):
  {"run_id": "FT-A01", "args": ["--mode", "partial", "--unfreeze", "blocks:2", "--epochs", "4", ...], "est_hours": 6.0,
   "max_gpu_other_gb": 35}
`run_id` is added to the train_ft command automatically.  Entry state is derived from the run directory
(results_ft/runs/<run_id>/STATUS.json), not stored in the queue file:
  no STATUS            -> pending
  state running (pid alive) -> running;   running (pid dead) -> interrupted (auto-resume)
  finished / refused / failed* / stopped_storage_guard -> done / not retried (failed: up to 2 auto-resumes)
  stopped (SIGTERM / request) -> stopped (resumed only by `resume-all`, not automatically, unless supervisor stopped it for a deadline)

CLI:  runqueue.py list | enqueue --est-hours H RUN_ID -- <train_ft args...> | resume-all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ckpt as C

ROOT = Path("/home/yejz1/experimental_research")
FT_DIR = ROOT / "research_management/finetuning"
QUEUE = FT_DIR / "QUEUE.json"
RUNS = C.RUNS_ROOT


def pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # zombie check
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except Exception:
        return False


def load_queue():
    try:
        q = json.loads(QUEUE.read_text())
    except FileNotFoundError:
        return []
    if isinstance(q, dict):
        q = q.get("runs", [])
    return q


def run_state(run_id: str) -> str:
    st = C.read_status(RUNS / run_id)
    if st is None:
        return "pending"
    s = st["state"]
    if s == "running":
        return "running" if pid_alive(st.get("pid")) else "interrupted"
    return s


def attempts(run_id: str) -> int:
    f = RUNS / run_id / "attempts.json"
    try:
        return json.loads(f.read_text())["n"]
    except Exception:
        return 0


def next_runnable(queue=None):
    """First queue entry that is pending or interrupted (auto-resume), skipping entries that failed twice."""
    for e in (queue if queue is not None else load_queue()):
        s = run_state(e["run_id"])
        if s in ("pending", "interrupted") and attempts(e["run_id"]) < 3:
            return e, s
    return None, None


def resumable_runs():
    """All run dirs that are not finished and have a valid-looking checkpoint dir: for resume-all."""
    out = []
    for d in sorted(RUNS.glob("*")):
        if not (d / "run_cmd.json").exists() or not list((d / "ckpts").glob("ckpt_step*.pt")):
            continue
        s = run_state(d.name)
        if s in ("interrupted", "stopped"):
            out.append(d.name)
    return out


def cmd_list():
    for e in load_queue():
        print(f"{e['run_id']:24s} {run_state(e['run_id']):12s} est {e.get('est_hours', '?')} h  {' '.join(e['args'])[:100]}")


def cmd_enqueue(a):
    q = load_queue()
    if any(e["run_id"] == a.run_id for e in q):
        sys.exit(f"{a.run_id} already queued")
    q.append(dict(run_id=a.run_id, args=a.train_args, est_hours=a.est_hours))
    C.atomic_write_text(QUEUE, json.dumps(q, indent=1))
    print("queued", a.run_id)


def cmd_resume_all():
    """Mark every interrupted/stopped run with checkpoints for resumption (supervisor launches them when GPU/deadline allow)."""
    rs = resumable_runs()
    for r in rs:
        (RUNS / r / "RESUME_ME").write_text("1")
        (RUNS / r / "REQUEST_STOP").unlink(missing_ok=True)
    print("marked for resume:", rs or "none")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("list")
    e = sp.add_parser("enqueue"); e.add_argument("run_id"); e.add_argument("--est-hours", type=float, required=True)
    sp.add_parser("resume-all")
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:                       # everything after `--` is passed verbatim to train_ft.py
        i = argv.index("--"); argv, extra = argv[:i], argv[i + 1:]
    a = ap.parse_args(argv)
    if a.cmd == "enqueue":
        a.train_args = extra
    {"list": cmd_list, "resume-all": cmd_resume_all}.get(a.cmd, lambda: cmd_enqueue(a))()
