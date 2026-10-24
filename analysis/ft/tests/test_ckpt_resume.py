"""Checkpoint/resume tests with the REAL model (partial mode, last transformer block + head), 12 train tiles, 3-step checkpoints.
Run: ./torch-venv/bin/python -m pytest -q -s analysis/ft/tests/test_ckpt_resume.py     (uses the GPU for ~2-3 min)
Scenarios: kill -9 + resume vs uninterrupted reference; SIGTERM graceful path; storage guard (fake usage env); corrupted newest
checkpoint fallback; config-hash incompatibility; storage_check arithmetic."""
import csv, hashlib, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path
import numpy as np, pytest, torch
ROOT = Path("/home/yejz1/experimental_research")
sys.path.insert(0, str(ROOT / "analysis/ft"))
import ckpt as C
import train_ft as T

PY = str(ROOT / "torch-venv/bin/python")
TRAIN = str(ROOT / "analysis/ft/train_ft.py")
OUT = ROOT / "results_ft/test_runs"
COMMON = ["--out-root", str(OUT), "--mode", "partial", "--unfreeze", "blocks:1", "--train-limit", "12", "--val-limit", "6",
          "--val-subset-n", "4", "--epochs", "2", "--ckpt-every-steps", "3", "--warmup-steps", "2", "--num-workers", "2",
          "--deterministic", "--precision", "fp32", "--log-every", "1", "--ckpt-every-min", "999"]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def launch(run, extra=(), env=None):
    e = dict(os.environ, **(env or {}))
    return subprocess.Popen([PY, TRAIN, "--run-id", run, *COMMON, *extra], env=e, start_new_session=True,
                            stdout=open(OUT / f"{run}.log", "a"), stderr=subprocess.STDOUT)


def metrics(run):
    f = OUT / run / "metrics.csv"
    return list(csv.DictReader(open(f))) if f.exists() else []


def last_losses(run):
    d = {}
    for r in metrics(run):
        d[int(r["global_step"])] = float(r["loss"])      # later rows (after resume) overwrite
    return d


def wait_for(cond, timeout=600, poll=0.2):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(poll)
    return False


def load_ck(run, step):
    return torch.load(OUT / run / "ckpts" / f"ckpt_step{step:08d}.pt", map_location="cpu", weights_only=False)


def opt_sha(sd):
    class S:  # shim with state_dict()
        def state_dict(self): return sd
    return T._opt_hash(S())


@pytest.fixture(scope="module", autouse=True)
def clean():
    shutil.rmtree(OUT, ignore_errors=True); OUT.mkdir(parents=True)
    yield


def test_kill9_resume_matches_reference():
    ref = launch("ref", ["--max-steps", "9"])
    assert ref.wait(timeout=600) == 0
    p = launch("k9", ["--max-steps", "9"], env={"FT_TEST_SLEEP_PER_STEP": "1.0"})
    assert wait_for(lambda: len(metrics("k9")) >= 7), "run did not reach step 7"
    os.killpg(p.pid, signal.SIGKILL); p.wait()
    ck_dir = OUT / "k9" / "ckpts"
    names = sorted(x.name for x in ck_dir.glob("ckpt_step*.pt"))
    assert names[-1] == "ckpt_step00000006.pt", names          # step 7 lost, step 6 checkpoint intact
    q = launch("k9", ["--max-steps", "9"])
    assert q.wait(timeout=600) == 0
    rv = json.loads((OUT / "k9" / "resume_verify.json").read_text())
    assert rv["global_step"] == 6 and rv["ckpt"] == "ckpt_step00000006.pt"
    c6 = load_ck("k9", 6)
    assert rv["opt_state_sha"] == opt_sha(c6["optimizer"]), "optimizer state restored != checkpoint"
    # reproducibility of the pre-kill trajectory and of the resumed continuation
    lk, lr = last_losses("k9"), last_losses("ref")
    print("\nstep losses ref :", {s: round(lr[s], 6) for s in range(1, 10)})
    print("step losses kill:", {s: round(lk[s], 6) for s in range(1, 10)})
    diffs = [abs(lk[s] - lr[s]) for s in range(1, 10)]
    print("max |loss diff| over steps 1-9 =", max(diffs), "; steps 7-9 (after resume) =", max(diffs[6:]))
    assert max(diffs[6:]) < 1e-5 and max(diffs[:6]) < 1e-5
    # final optimizer states / params at step 9
    a, b = load_ck("k9", 9), load_ck("ref", 9)
    assert a["state"]["global_step"] == b["state"]["global_step"] == 9
    mx = 0.0
    for k in a["optimizer"]["state"]:
        for kk in ("exp_avg", "exp_avg_sq"):
            mx = max(mx, float((a["optimizer"]["state"][k][kk] - b["optimizer"]["state"][k][kk]).abs().max()))
    pm = max(float((a["trainable"][k] - b["trainable"][k]).abs().max()) for k in a["trainable"])
    print(f"max |opt state diff| at step 9 = {mx:.3e}; max |trainable param diff| = {pm:.3e}")
    assert mx < 1e-6 and pm < 1e-6
    assert a["state"]["epoch_pos"] == b["state"]["epoch_pos"] == 9 and a["rng"]["torch"].shape == b["rng"]["torch"].shape
    for k in ("meta", "config_hash", "format"):
        assert k in a
    assert set(a["meta"]) >= {"git_commit", "versions", "split_hash", "frozen_source", "arch_cfg"}
    # only trainable params stored
    assert all(k.startswith(("head.", "trunk.transformer.7.")) for k in a["trainable"]), list(a["trainable"])[:3]
    assert len(sorted((OUT / "k9" / "ckpts").glob("ckpt_step*.pt"))) == 2


def test_kill9_resume_bf16_autocast():
    """Same as above with bf16 autocast (deterministic kernels): tolerance 1e-3 on the loss, 1e-5 on params."""
    ex = ["--max-steps", "9", "--precision", "bf16"]
    ref = launch("ref16", ex); assert ref.wait(timeout=600) == 0
    p = launch("k16", ex, env={"FT_TEST_SLEEP_PER_STEP": "1.0"})
    assert wait_for(lambda: len(metrics("k16")) >= 8)
    os.killpg(p.pid, signal.SIGKILL); p.wait()
    q = launch("k16", ex); assert q.wait(timeout=600) == 0
    rv = json.loads((OUT / "k16" / "resume_verify.json").read_text())
    assert rv["global_step"] == 6
    lk, lr = last_losses("k16"), last_losses("ref16")
    d = max(abs(lk[s] - lr[s]) for s in range(1, 10))
    a, b = load_ck("k16", 9), load_ck("ref16", 9)
    pm = max(float((a["trainable"][k] - b["trainable"][k]).abs().max()) for k in a["trainable"])
    print(f"\nbf16: max |loss diff| steps 1-9 = {d:.3e}; max |param diff| step 9 = {pm:.3e}")
    assert d < 1e-3 and pm < 1e-5


def test_sigterm_graceful_and_resume():
    p = launch("term", ["--max-steps", "9"], env={"FT_TEST_SLEEP_PER_STEP": "1.0"})
    assert wait_for(lambda: len(metrics("term")) >= 4)
    p.send_signal(signal.SIGTERM)
    rc = p.wait(timeout=120)
    st = C.read_status(OUT / "term")
    steps = sorted(int(x.name[-11:-3]) for x in (OUT / "term" / "ckpts").glob("ckpt_step*.pt"))
    print("\nSIGTERM: exit", rc, "status", st["state"], "ckpt steps", steps)
    assert rc == 7 and st["state"] == "stopped"
    last = int(metrics("term")[-1]["global_step"])
    assert steps[-1] >= last - 0 and steps[-1] in (last, last + 1)
    q = launch("term", ["--max-steps", "9"])
    assert q.wait(timeout=600) == 0
    rv = json.loads((OUT / "term" / "resume_verify.json").read_text())
    assert rv["global_step"] == steps[-1]
    assert max(last_losses("term")) == 9
    assert abs(last_losses("term")[9] - last_losses("ref")[9]) < 1e-5     # same trajectory as uninterrupted reference


def test_storage_guard_clean_stop():
    # first 6 steps normally (ckpt@3, @6) then simulate full quota for the next checkpoint
    shutil.rmtree(OUT / "guard", ignore_errors=True)
    p = launch("guard", ["--max-steps", "6"]); assert p.wait(timeout=600) == 0
    before = {x.name: x.stat().st_mtime_ns for x in (OUT / "guard" / "ckpts").glob("ckpt_step*.pt")}
    p = launch("guard", ["--max-steps", "12", "--epochs", "4"], env={"FT_STORAGE_FAKE_USAGE_GB": "178"})
    rc = p.wait(timeout=600)
    st = C.read_status(OUT / "guard")
    after = {x.name: x.stat().st_mtime_ns for x in (OUT / "guard" / "ckpts").glob("ckpt_step*.pt")}
    print("\nstorage guard: exit", rc, st["state"], st.get("reason"))
    assert rc == 4 and st["state"] == "stopped_storage_guard"
    assert before == after, "previous checkpoints must be untouched"
    assert not list((OUT / "guard" / "ckpts").glob(".*tmp*"))


def test_storage_check_arithmetic(monkeypatch):
    monkeypatch.setenv("FT_STORAGE_FAKE_USAGE_GB", "170")
    C.storage_check(1e9)                                   # 186.3-170-1 = 15.3 left: ok
    with pytest.raises(C.StorageGuardError):
        C.storage_check(13e9)                              # 12.1 GiB -> leaves 4.2 GB < 5
    monkeypatch.setenv("FT_STORAGE_FAKE_USAGE_GB", "175.5")
    with pytest.raises(C.StorageGuardError):
        C.storage_check(1e6)                               # usage above 175 GB
    monkeypatch.delenv("FT_STORAGE_FAKE_USAGE_GB")
    u, q, src = C.home_usage_gb()
    print("\nreal home usage:", u, q, src)
    assert src == "accre_storage" and 0 < u < q


def test_corrupt_newest_and_config_mismatch():
    run = OUT / "k9"
    ck_dir = run / "ckpts"
    files = sorted(ck_dir.glob("ckpt_step*.pt"))
    assert len(files) == 2
    p = json.loads((run / "run_cmd.json").read_text())
    sys.argv = ["x"] + p["argv"]
    args = T.parse(); cfgd = vars(args)
    mgr = C.CheckpointManager(run, cfgd, {}, {}, T.D.tiles_hash())
    f, payload = mgr.find_latest_valid()
    assert f.name == "ckpt_step00000009.pt"
    # truncate the newest (simulated crash during a non-atomic write): must fall back to the previous one
    sz = files[-1].stat().st_size
    with open(files[-1], "r+b") as fh:
        fh.truncate(sz // 2)
    f, payload = mgr.find_latest_valid()
    assert f.name == "ckpt_step00000006.pt" and payload["state"]["global_step"] == 6
    # incompatible config (different lr) -> nothing valid
    cfg2 = dict(cfgd, lr_head=0.5)
    mgr2 = C.CheckpointManager(run, cfg2, {}, {}, T.D.tiles_hash())
    assert mgr2.find_latest_valid() == (None, None)
    # no leftover tmp files from atomic saves
    assert not list(ck_dir.glob(".*tmp*"))


def test_sigusr1_and_request_file_checkpoint_and_continue():
    ex = ["--max-steps", "8", "--ckpt-every-steps", "0"]
    p = launch("usr1", ex, env={"FT_TEST_SLEEP_PER_STEP": "1.0"})
    assert wait_for(lambda: len(metrics("usr1")) >= 2)
    p.send_signal(signal.SIGUSR1)
    assert wait_for(lambda: list((OUT / "usr1" / "ckpts").glob("ckpt_step*.pt")), timeout=30), "no checkpoint after SIGUSR1"
    first = sorted((OUT / "usr1" / "ckpts").glob("ckpt_step*.pt"))[-1].name
    assert p.poll() is None, "SIGUSR1 must not stop the run"
    (OUT / "usr1" / "REQUEST_CKPT").write_text("1")
    assert wait_for(lambda: len(list((OUT / "usr1" / "ckpts").glob("ckpt_step*.pt"))) >= 2, timeout=30)
    assert not (OUT / "usr1" / "REQUEST_CKPT").exists()
    assert p.wait(timeout=300) == 0
    print("\nUSR1 checkpoint:", first, "-> run continued to finish, status", C.read_status(OUT / "usr1")["state"])
