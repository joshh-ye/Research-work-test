"""Deadline supervisor + run launcher for fine-tuning.  Start (independent of any Claude session):
    setsid nohup ./torch-venv/bin/python analysis/ft/supervisor.py > research_management/finetuning/supervisor.out 2>&1 &
(analysis/ft/resume_all.sh does this and is idempotent via supervisor.pid.)

Every `--interval` s (300 default) it
  * reads deadline.json (milestones: deadline_awareness Mon 16:00, no_new_expensive Mon 18:00, final_consolidation 20:00,
    artifacts_saved 21:30, hard_stop 22:00 CDT) and writes research_management/finetuning/supervisor_status.json
    (time, remaining h, runs + PIDs + last-checkpoint age + progress, GPU memory, home usage) ;
  * appends a line to SUPERVISOR_LOG.md every `--log-every` s (900 default: 15 min);
  * writes LAUNCH_POLICY.json and the flag file NO_NEW_RUNS (after artifacts_saved);
  * enforces:  >= no_new_expensive: fresh runs launched only if est_hours <= time left to artifacts_saved (policy.launch_allowed,
      also checked inside train_ft.py itself);   >= final_consolidation (20:00): SIGUSR1 to every running run (forces a checkpoint;
      training continues) and SIGTERM/REQUEST_STOP to runs that cannot reach another scheduled checkpoint before 21:30
      (now + ckpt interval > 21:30);   >= artifacts_saved (21:30): SIGTERM (+REQUEST_STOP) to ALL runs (train_ft writes a final
      checkpoint then exits, code 7); SIGKILL only if a run is still alive 20 min later (last valid checkpoint stays);
      NO training is started at/after 21:30;
  * queue: if no live run and GPU memory used < threshold (other engineer's jobs), launches (1) runs marked RESUME_ME (resume-all),
      (2) interrupted queue entries (state running with dead pid -> automatic resume from find_latest_valid), (3) the next pending
      entry of QUEUE.json whose estimated duration passes the deadline policy.  One training run at a time.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ckpt as C
import policy as P
import runqueue as Q

ROOT = Path("/home/yejz1/experimental_research")
FT_DIR = P.FT_DIR
PY = str(ROOT / "torch-venv/bin/python")
TRAIN = str(HERE / "train_ft.py")
STATUS_JSON = FT_DIR / "supervisor_status.json"
LOG_MD = FT_DIR / "SUPERVISOR_LOG.md"
STATE_JSON = FT_DIR / "supervisor_state.json"
POLICY_JSON = FT_DIR / "LAUNCH_POLICY.json"
NO_NEW = FT_DIR / "NO_NEW_RUNS"
PIDFILE = FT_DIR / "supervisor.pid"
GPU_OTHER_MAX_GB = 35.0       # launch only if GPU memory in use (other engineer) is below this


def gpu_mem():
    try:
        o = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], text=True, timeout=20)
        u, t = [float(x) / 1024 for x in o.strip().split(",")]
        return u, t
    except Exception:
        return None, None


class Supervisor:
    def __init__(self, interval=300, log_every=900, dry=False):
        self.interval, self.log_every, self.dry = interval, log_every, dry
        self.children = {}          # run_id -> Popen
        self.last_log = 0.0
        try:
            self.state = json.loads(STATE_JSON.read_text())
        except Exception:
            self.state = {}
        self.state.setdefault("usr1_sent", {})
        self.state.setdefault("term_sent", {})
        self.actions = []

    # ---- helpers (overridable in tests)
    def send(self, pid, sig):
        self.actions.append(("signal", pid, int(sig)))
        if not self.dry:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass

    def launch(self, run_id, args, resume):
        self.actions.append(("launch", run_id, resume))
        if self.dry:
            return None
        rd = C.RUNS_ROOT / run_id
        rd.mkdir(parents=True, exist_ok=True)
        n = Q.attempts(run_id) + 1
        C.atomic_write_text(rd / "attempts.json", json.dumps(dict(n=n, last=time.strftime("%Y-%m-%dT%H:%M:%S"))))
        (rd / "RESUME_ME").unlink(missing_ok=True)
        (rd / "REQUEST_STOP").unlink(missing_ok=True)
        p = subprocess.Popen([PY, TRAIN, "--run-id", run_id, *args], cwd=ROOT, start_new_session=True,
                             stdout=open(rd / "stdout.log", "a"), stderr=subprocess.STDOUT)
        self.children[run_id] = p
        return p

    def running_runs(self):
        out = []
        for d in sorted(C.RUNS_ROOT.glob("*")):
            st = C.read_status(d)
            if st and st["state"] == "running" and Q.pid_alive(st.get("pid")):
                prog = {}
                try:
                    prog = json.loads((d / "progress.json").read_text())
                except Exception:
                    pass
                ck = sorted(d.glob("ckpts/ckpt_step*.pt"))
                last_ck = ck[-1].stat().st_mtime if ck else None
                out.append(dict(run_id=d.name, pid=st["pid"], global_step=prog.get("global_step"), epoch=prog.get("epoch"),
                                last_ckpt_age_min=(None if last_ck is None else round((time.time() - last_ck) / 60, 1)),
                                ckpt_every_s=prog.get("ckpt_every_s", 1800), est_remaining_h=(None if not prog.get("est_remaining_s") else round(prog["est_remaining_s"] / 3600, 2)),
                                s_per_step=prog.get("s_per_step_ema"), best_val=prog.get("best_val")))
        return out

    # ---- one supervision tick
    def tick(self, now=None):
        now = now or P.now_utc()
        self.actions = []
        for rid, p in list(self.children.items()):   # reap
            if p.poll() is not None:
                del self.children[rid]
        d, ms = P.load_deadline()
        rem_stop = P.hours_to(datetime.fromisoformat(d["effective_stop"]), now)
        left_art = P.hours_to(ms["artifacts_saved"], now)
        runs = self.running_runs()

        # --- deadline enforcement
        if now >= ms["final_consolidation"]:
            for r in runs:
                if r["run_id"] not in self.state["usr1_sent"] and now < ms["artifacts_saved"]:
                    self.send(r["pid"], signal.SIGUSR1)
                    self.state["usr1_sent"][r["run_id"]] = now.isoformat()
                    if now.timestamp() + r["ckpt_every_s"] > ms["artifacts_saved"].timestamp() and r["run_id"] not in self.state["term_sent"]:
                        # cannot reach another scheduled checkpoint before 21:30: stop it now (the USR1 checkpoint is on disk first;
                        # SIGTERM also forces a checkpoint at the next step boundary)
                        (C.RUNS_ROOT / r["run_id"] / "REQUEST_STOP").write_text("deadline")
                        self.send(r["pid"], signal.SIGTERM)
                        self.state["term_sent"][r["run_id"]] = now.isoformat()
        if now >= ms["artifacts_saved"]:
            for r in runs:
                first = self.state["term_sent"].get(r["run_id"])
                if first is None:
                    (C.RUNS_ROOT / r["run_id"] / "REQUEST_STOP").write_text("deadline-final")
                    self.send(r["pid"], signal.SIGTERM)
                    self.state["term_sent"][r["run_id"]] = now.isoformat()
                elif (now - datetime.fromisoformat(first)).total_seconds() > 20 * 60 and now.timestamp() >= ms["artifacts_saved"].timestamp() + 20 * 60:
                    self.send(r["pid"], signal.SIGKILL)   # last valid checkpoint stays on disk
        # --- policy files
        policy = dict(time=now.isoformat(), hours_to_artifacts_saved=round(left_art, 3), allow_new_runs=left_art > 0,
                      fresh_run_max_est_hours=(None if now < ms["no_new_expensive"] else max(0.0, round(left_art, 3))),
                      no_training_start=left_art <= 0)
        if not self.dry:
            C.atomic_write_text(POLICY_JSON, json.dumps(policy, indent=1))
            if left_art <= 0:
                NO_NEW.write_text(now.isoformat())
            else:
                NO_NEW.unlink(missing_ok=True)

        # --- launching
        used, total = gpu_mem()
        launched = None
        if left_art > 0 and not runs and not self.children:
            gpu_ok = used is None or used < GPU_OTHER_MAX_GB
            if gpu_ok:
                launched = self.maybe_launch(now)
        sess = dict(time=now.isoformat(), time_local=now.astimezone(datetime.fromisoformat(d["effective_stop"]).tzinfo).strftime("%a %Y-%m-%d %H:%M:%S %Z"),
                    remaining_h_to_effective_stop=round(rem_stop, 3), hours_to_artifacts_saved=round(left_art, 3), runs=runs,
                    gpu_used_gb=used, gpu_total_gb=total, gpu_other_max_gb_for_launch=GPU_OTHER_MAX_GB,
                    home_usage_gb=C.home_usage_gb()[:2], supervisor_pid=os.getpid(), queue=[(e["run_id"], Q.run_state(e["run_id"])) for e in Q.load_queue()],
                    actions=self.actions, launched=launched, policy=policy)
        if not self.dry:
            C.atomic_write_text(STATUS_JSON, json.dumps(sess, indent=1, default=str))
            C.atomic_write_text(STATE_JSON, json.dumps(self.state))
        return sess

    def maybe_launch(self, now):
        # 1) explicit resume requests
        for d in sorted(C.RUNS_ROOT.glob("*")):
            if (d / "RESUME_ME").exists() and (d / "run_cmd.json").exists() and Q.run_state(d.name) in ("interrupted", "stopped", "pending"):
                ok, why = P.launch_allowed(0, resume=True, now=now)
                if ok:
                    self.launch(d.name, _args_from_cmd(d), resume=True)
                    return d.name
        # 2/3) queue (interrupted first by position, then pending)
        for e in Q.load_queue():
            s = Q.run_state(e["run_id"])
            if s not in ("pending", "interrupted") or Q.attempts(e["run_id"]) >= 3:
                continue
            ok, why = P.launch_allowed(float(e.get("est_hours", 0.0)), resume=(s == "interrupted"), now=now)
            if not ok:
                self.actions.append(("skip", e["run_id"], why))
                continue
            self.launch(e["run_id"], e["args"], resume=(s == "interrupted"))
            return e["run_id"]
        return None

    def log_line(self, s):
        if time.time() - self.last_log < self.log_every:
            return
        self.last_log = time.time()
        if not LOG_MD.exists():
            LOG_MD.write_text("# Supervisor log (appended every 15 min; analysis/ft/supervisor.py)\n\n| local time | h to stop | h to 21:30 | runs (id pid step ckpt-age min) | GPU used GB | home GB | actions |\n|---|---|---|---|---|---|---|\n")
        rs = "; ".join(f"{r['run_id']} pid{r['pid']} step{r['global_step']} ck{r['last_ckpt_age_min']}m" for r in s["runs"]) or "none"
        with open(LOG_MD, "a") as f:
            f.write(f"| {s['time_local']} | {s['remaining_h_to_effective_stop']:.2f} | {s['hours_to_artifacts_saved']:.2f} | {rs} | "
                    f"{s['gpu_used_gb'] if s['gpu_used_gb'] is None else round(s['gpu_used_gb'], 1)} | {s['home_usage_gb'][0]:.1f} | "
                    f"{'; '.join(map(str, s['actions'])) or '-'} |\n")


def _args_from_cmd(d: Path):
    argv = json.loads((d / "run_cmd.json").read_text())["argv"]
    # drop --run-id X (re-added by launch())
    out, i = [], 0
    while i < len(argv):
        if argv[i] == "--run-id":
            i += 2
            continue
        out.append(argv[i]); i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--log-every", type=int, default=900)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    if PIDFILE.exists():
        try:
            old = int(PIDFILE.read_text())
            if old != os.getpid() and Q.pid_alive(old) and "supervisor.py" in Path(f"/proc/{old}/cmdline").read_text():
                sys.exit(f"supervisor already running (pid {old})")
        except (ValueError, FileNotFoundError, ProcessLookupError):
            pass
    PIDFILE.write_text(str(os.getpid()))
    sup = Supervisor(a.interval, a.log_every)
    stop = {"v": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("v", True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("v", True))
    print(f"supervisor pid {os.getpid()} started {time.strftime('%F %T')}", flush=True)
    while not stop["v"]:
        try:
            s = sup.tick()
            sup.log_line(s)
        except Exception as e:  # never die on a transient error
            print("tick error:", repr(e), flush=True)
        if a.once:
            break
        t_end = time.time() + a.interval
        while time.time() < t_end and not stop["v"]:
            time.sleep(2)
    PIDFILE.unlink(missing_ok=True)
    print("supervisor exiting", flush=True)


if __name__ == "__main__":
    main()
