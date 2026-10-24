"""Deadline policy shared by train_ft.py, supervisor.py and the queue (reads research_management/finetuning/deadline.json).
All times tz-aware (America/Chicago offsets are in the json)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/yejz1/experimental_research")
FT_DIR = ROOT / "research_management/finetuning"
DEADLINE_JSON = FT_DIR / "deadline.json"


def load_deadline():
    d = json.loads(DEADLINE_JSON.read_text())
    ms = {k: datetime.fromisoformat(v) for k, v in d["milestones"].items()}
    return d, ms


def now_utc():
    return datetime.now(timezone.utc)


def hours_to(ts, now=None):
    return (ts - (now or now_utc())).total_seconds() / 3600.0


def launch_allowed(est_hours: float, resume: bool = False, now=None):
    """-> (allowed, reason). Fresh runs: after `no_new_expensive` (Mon 18:00 CDT) only if est_hours <= time left until
    `artifacts_saved` (21:30 CDT).  Resumes of interrupted runs: allowed until `artifacts_saved`.  Nothing starts after 21:30."""
    d, ms = load_deadline()
    now = now or now_utc()
    left = hours_to(ms["artifacts_saved"], now)
    if left <= 0:
        return False, "after artifacts_saved (21:30 CDT): no training may start"
    if resume:
        return True, f"resume allowed ({left:.2f} h to artifacts_saved)"
    if now >= ms["no_new_expensive"] and est_hours > left:
        return False, f"after no_new_expensive: est {est_hours:.2f} h > {left:.2f} h left to artifacts_saved"
    return True, f"ok ({left:.2f} h to artifacts_saved)"
