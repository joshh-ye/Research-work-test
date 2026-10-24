"""Deadline-supervisor logic with simulated clock (dry mode: no signals sent, no processes launched). Run: pytest -q analysis/ft/tests/test_supervisor.py"""
import json, subprocess, sys, signal
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest
ROOT = Path("/home/yejz1/experimental_research")
sys.path.insert(0, str(ROOT / "analysis/ft"))
import ckpt as C, runqueue as Q, supervisor as S, policy as P

CDT = timezone(timedelta(hours=-5))


def at(h, m=0, day=26):
    return datetime(2026, 10, day, h, m, tzinfo=CDT)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "RUNS_ROOT", tmp_path / "runs"); (tmp_path / "runs").mkdir()
    monkeypatch.setattr(Q, "RUNS", tmp_path / "runs"); monkeypatch.setattr(Q, "QUEUE", tmp_path / "QUEUE.json")
    monkeypatch.setattr(S, "gpu_mem", lambda: (5.0, 80.0))
    monkeypatch.setenv("FT_STORAGE_FAKE_USAGE_GB", "100")
    sleeper = subprocess.Popen(["sleep", "120"])
    rd = tmp_path / "runs" / "R1"; rd.mkdir()
    C.write_status(rd, "running"); st = json.loads((rd / "STATUS.json").read_text()); st["pid"] = sleeper.pid
    (rd / "STATUS.json").write_text(json.dumps(st))
    (rd / "progress.json").write_text(json.dumps(dict(global_step=5, ckpt_every_s=1800, est_remaining_s=3600)))
    yield tmp_path, sleeper
    sleeper.kill(); sleeper.wait()


def kinds(s): return [a for a in s.actions if a[0] == "signal"]


def test_before_20_no_signals(env):
    s = S.Supervisor(dry=True); s.tick(at(19, 59))
    assert kinds(s) == []


def test_20_00_usr1_and_no_stop_when_ckpt_reachable(env):
    s = S.Supervisor(dry=True); r = s.tick(at(20, 1))
    pid = env[1].pid
    assert ("signal", pid, int(signal.SIGUSR1)) in s.actions and ("signal", pid, int(signal.SIGTERM)) not in s.actions
    assert r["runs"][0]["pid"] == pid


def test_20_00_stop_run_that_cannot_reach_another_checkpoint(env):
    rd = env[0] / "runs" / "R1"; (rd / "progress.json").write_text(json.dumps(dict(ckpt_every_s=7200)))
    s = S.Supervisor(dry=True); s.tick(at(20, 1))
    assert ("signal", env[1].pid, int(signal.SIGTERM)) in s.actions


def test_21_30_sigterm_all_and_no_launch(env):
    (env[0] / "QUEUE.json").write_text(json.dumps([dict(run_id="Q1", args=["--mode", "frozen"], est_hours=0.1)]))
    s = S.Supervisor(dry=True); r = s.tick(at(21, 31))
    assert ("signal", env[1].pid, int(signal.SIGTERM)) in s.actions
    assert not [a for a in s.actions if a[0] == "launch"] and r["policy"]["no_training_start"]
    s.state["term_sent"]["R1"] = at(21, 31).isoformat()
    s.tick(at(21, 55)); assert ("signal", env[1].pid, int(signal.SIGKILL)) in s.actions        # 24 min after SIGTERM


def test_queue_policy_after_18(env):
    sleeper = env[1]; sleeper.kill(); sleeper.wait()                       # GPU free: no live run
    (env[0] / "QUEUE.json").write_text(json.dumps([dict(run_id="LONG", args=["--mode", "full"], est_hours=5.0),
                                                   dict(run_id="SHORT", args=["--mode", "frozen"], est_hours=2.0)]))
    s = S.Supervisor(dry=True); s.tick(at(18, 30))          # 3.0 h left to 21:30
    assert ("launch", "SHORT", False) in s.actions and not [a for a in s.actions if a[:2] == ("launch", "LONG")]
    assert any(a[0] == "skip" and a[1] == "LONG" for a in s.actions)
    s2 = S.Supervisor(dry=True); s2.tick(at(17, 0))         # before 18:00 everything allowed, queue order respected
    assert ("launch", "LONG", False) in s2.actions


def test_interrupted_run_auto_resume(env):
    sleeper = env[1]; sleeper.kill(); sleeper.wait()                       # run dir says running, pid dead -> interrupted
    (env[0] / "QUEUE.json").write_text(json.dumps([dict(run_id="R1", args=["--mode", "frozen"], est_hours=9.0)]))
    assert Q.run_state("R1") == "interrupted"
    s = S.Supervisor(dry=True); s.tick(at(19, 0))          # est 9 h > left, but a resume is allowed
    assert ("launch", "R1", True) in s.actions


def test_policy_function():
    assert P.launch_allowed(1.0, now=at(21, 31))[0] is False
    assert P.launch_allowed(4.0, now=at(18, 0))[0] is False and P.launch_allowed(3.4, now=at(18, 0))[0] is True
    assert P.launch_allowed(100.0, resume=True, now=at(21, 29))[0] is True
