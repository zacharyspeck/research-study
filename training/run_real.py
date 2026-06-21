"""Thin, defensive wrapper that the 12 real runs (2x2 conditions x 3 seeds) share.

It wraps the vetted `train_one` (never modifies it): deterministic per-run private
HF repos, idempotent skip-if-done, NaN/OOM caught and recorded as a run status,
incremental reward flushing so a crash keeps the partial trend, a fixed-count
re-assert, and a results.json saved alongside the adapter off-session.

Pure helpers are import-light and unit-testable; torch / huggingface_hub are
deferred into the functions that need them (no GPU needed to import this module).
"""

from __future__ import annotations

# Memory-fragmentation fix for the 16GB T4 — set BEFORE torch is imported anywhere
# (here at module import, so every entry point gets it without a manual Kaggle cell).
# expandable_segments lets the CUDA allocator return freed blocks, avoiding the
# fragmentation OOM the full-config calibration run hit. None of the imports below
# pull in torch (train_grpo defers it), so this executes before torch initializes.
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import re
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import study_config as cfg
from data_generation.generate import build_all
from training.train_grpo import NaNLossError, train_one

GPU_HOURS_PER_WEEK = 30.0      # free-tier weekly budget, for the extrapolation
N_TOTAL_RUNS = 12              # 2x2 conditions x 3 seeds

# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested in tests/test_real.py).                           #
# --------------------------------------------------------------------------- #
def repo_id(reward_mode, difficulty, seed, user):
    """Deterministic, unique private-repo name per (condition, seed)."""
    return f"{user}/rlvr-taskA-{reward_mode}-{difficulty}-seed{seed}"


def reward_trended_up(reward_history):
    """First-quarter vs last-quarter mean reward (pure).

    Returns {first_q_mean, last_q_mean, delta, max_reward, trended_up}.
    `trended_up` is last_q_mean > first_q_mean. Short/empty histories don't crash.
    """
    rewards = [h["reward"] for h in reward_history
               if isinstance(h, dict) and h.get("reward") is not None]
    n = len(rewards)
    if n == 0:
        return {"first_q_mean": None, "last_q_mean": None, "delta": None,
                "max_reward": None, "trended_up": False}
    q = max(1, n // 4)
    first = rewards[:q]
    last = rewards[-q:]
    first_q_mean = sum(first) / len(first)
    last_q_mean = sum(last) / len(last)
    return {
        "first_q_mean": first_q_mean,
        "last_q_mean": last_q_mean,
        "delta": last_q_mean - first_q_mean,
        "max_reward": max(rewards),
        "trended_up": last_q_mean > first_q_mean,
    }


def missing_runs(done_repos, expected):
    """Set of expected runs not yet done (pure)."""
    return set(expected) - set(done_repos)


def build_results(reward_mode, difficulty, seed, status, commit, runtime_sec,
                  peak_mem_gb, reward_history, reward_trend, adapter_repo, config_snapshot):
    """Assemble the results.json payload (pure)."""
    return {
        "reward_mode": reward_mode,
        "difficulty": difficulty,
        "seed": seed,
        "status": status,
        "commit": commit,
        "config_snapshot": config_snapshot,
        "runtime_sec": runtime_sec,
        "peak_mem_gb": peak_mem_gb,
        "reward_history": reward_history,
        "reward_trend": reward_trend,
        "adapter_repo": adapter_repo,
    }


def _parse_reward_line(line):
    """Parse a `step N: reward mean=X std=Y` log line into a record, or None."""
    m = re.search(r"step\s+(\d+):\s*reward mean=([-+0-9.eE]+)\s+std=(\S+)", line)
    if not m:
        return None
    try:
        std = float(m.group(3))
    except ValueError:
        std = None
    return {"step": int(m.group(1)), "reward": float(m.group(2)), "reward_std": std}


def _fmt_hm(seconds):
    """Seconds -> 'H:MM'."""
    seconds = int(seconds)
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}"


def _classify_error(exc):
    """Map a caught training exception to a run status string."""
    if isinstance(exc, NaNLossError):
        return "nan"
    msg = str(exc).lower()
    if "out of memory" in msg or type(exc).__name__ == "OutOfMemoryError":
        return "oom"
    return "error"


# --------------------------------------------------------------------------- #
# Incremental reward flushing — wrap train_one's stdout from the wrapper side  #
# (train_one is not modified) so a mid-run crash keeps the partial trend.      #
# --------------------------------------------------------------------------- #
class _RewardTee:
    """Mirror stdout, and flush parsed `reward mean=` log lines to a JSONL as they print."""

    def __init__(self, jsonl_path):
        self._orig = sys.stdout
        self._path = Path(jsonl_path)
        self._fh = None
        self.history = []

    def __enter__(self):
        self._orig = sys.stdout
        self._fh = open(self._path, "a", encoding="utf-8")
        sys.stdout = self
        return self

    def __exit__(self, *exc):
        sys.stdout = self._orig
        if self._fh:
            self._fh.close()
        return False

    def write(self, s):
        self._orig.write(s)
        for line in s.splitlines():
            rec = _parse_reward_line(line)
            if rec is not None:
                self.history.append(rec)
                self._fh.write(json.dumps(rec) + "\n")
                self._fh.flush()
        return len(s)

    def flush(self):
        self._orig.flush()

    def __getattr__(self, name):
        return getattr(self._orig, name)


# --------------------------------------------------------------------------- #
# Helpers that touch git / data / the Hub.                                    #
# --------------------------------------------------------------------------- #
def _git_short_sha():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True, cwd=_REPO_ROOT)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _expected_item_count(difficulty, seed):
    """Re-derive the item count a run uses (mirrors train_one's slicing) for the control."""
    data = build_all(seed, cfg.N_EASY, cfg.N_HARD, cfg.N_OOD_EASY, cfg.N_OOD_HARD)
    n = cfg.TRAIN_EXAMPLES
    if difficulty == "easy":
        return len(data["train_easy"][:n])
    half = n // 2
    return len(data["train_easy"][:half] + data["train_hard"][:half])


def _hf_token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _try_load_repo_results(repo, token):
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo, filename="results.json", token=token, repo_type="model")
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _upload_results(repo, results_path, token):
    from huggingface_hub import upload_file
    upload_file(path_or_fileobj=str(results_path), path_in_repo="results.json",
                repo_id=repo, repo_type="model", token=token)
    print(f"results.json pushed → {repo}")


def _print_readout(results, repo):
    trend = results["reward_trend"]
    runtime_sec = results["runtime_sec"]
    hours = runtime_sec / 3600.0
    if hours > 0:
        runs_per_week = GPU_HOURS_PER_WEEK / hours
        weeks_for_all = N_TOTAL_RUNS * hours / GPU_HOURS_PER_WEEK
        budget = (f"{hours:.2f} GPU-hr → ~{runs_per_week:.0f} runs/wk at {GPU_HOURS_PER_WEEK:.0f} "
                  f"GPU-hr/wk → {N_TOTAL_RUNS} runs ≈ {weeks_for_all:.1f} weeks")
    else:
        budget = "n/a"
    print("-" * 64)
    print(f"{results['reward_mode']}/{results['difficulty']} seed{results['seed']}  "
          f"| status={results['status']}")
    print(f"runtime: {_fmt_hm(runtime_sec)}  | peak mem: {results['peak_mem_gb']:.2f} GB")
    print(f"reward trend: first-q {trend['first_q_mean']} → last-q {trend['last_q_mean']}  "
          f"| trended_up={trend['trended_up']}")
    print(f"budget: {budget}")
    print(f"SAVED ✅ → {repo}")


# --------------------------------------------------------------------------- #
# The reusable real run (torch / HF deferred inside).                         #
# --------------------------------------------------------------------------- #
def run_cell(reward_mode, difficulty, seed, out_root="/kaggle/working/runs"):
    """One real run at the FULL frozen config (NO SMOKE overrides)."""
    import torch
    from huggingface_hub import repo_exists, whoami

    if reward_mode not in ("strict", "loose"):
        raise ValueError(f"reward_mode must be 'strict' or 'loose', got {reward_mode!r}")
    if difficulty not in ("easy", "easy_hard"):
        raise ValueError(f"difficulty must be 'easy' or 'easy_hard', got {difficulty!r}")

    token = _hf_token()
    if not token:
        raise RuntimeError(
            "An HF token is REQUIRED for the real runs — the adapter and results.json must be saved "
            "off-session before Kaggle wipes local files. Set HF_TOKEN (write scope) and re-run."
        )
    user = whoami(token=token)["name"]
    repo = repo_id(reward_mode, difficulty, seed, user)

    local_dir = Path(out_root) / f"{reward_mode}-{difficulty}-seed{seed}"
    local_dir.mkdir(parents=True, exist_ok=True)

    # Idempotency: a finished run already lives on the Hub -> never redo or clobber it.
    if repo_exists(repo, token=token):
        print(f"already complete — skipping {repo}")
        existing = _try_load_repo_results(repo, token)
        if existing is not None:
            existing["skipped"] = True
            return existing
        return {
            "reward_mode": reward_mode, "difficulty": difficulty, "seed": seed,
            "status": "skipped", "skipped": True, "commit": _git_short_sha(),
            "runtime_sec": 0.0, "peak_mem_gb": 0.0, "reward_history": [],
            "reward_trend": reward_trended_up([]), "adapter_repo": repo, "repo": repo,
        }

    commit = _git_short_sha()
    start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    reward_jsonl = local_dir / "reward_history.jsonl"
    if reward_jsonl.exists():
        reward_jsonl.unlink()   # fresh incremental log for this attempt

    status, error_msg, result = "ok", None, None
    with _RewardTee(reward_jsonl) as tee:
        try:
            result = train_one(reward_mode, difficulty, seed,
                               str(local_dir / "adapter"), save_repo=repo)
        except Exception as exc:   # NaN-stop, CUDA OOM, or anything else -> recorded, not crashed
            status = _classify_error(exc)
            error_msg = f"{type(exc).__name__}: {exc}"
    # On success use train_one's canonical history; on crash use the flushed partial.
    history = (result or {}).get("reward_history") or tee.history

    runtime_sec = time.time() - start
    peak_mem_gb = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0

    # Fixed-count control: this condition must train on exactly TRAIN_EXAMPLES items.
    items_used = _expected_item_count(difficulty, seed)
    assert items_used == cfg.TRAIN_EXAMPLES, (
        f"fixed-count control FAILED: {reward_mode}/{difficulty} used {items_used} items, "
        f"expected {cfg.TRAIN_EXAMPLES}."
    )

    trend = reward_trended_up(history)
    adapter_repo = (result or {}).get("adapter_repo") or (repo if status == "ok" else None)
    results = build_results(reward_mode, difficulty, seed, status, commit, runtime_sec,
                            peak_mem_gb, history, trend, adapter_repo, cfg.snapshot())
    results["error"] = error_msg
    results["items_used"] = items_used
    results["repo"] = repo

    # Always save locally. Push results.json to the repo ONLY on success, so the
    # idempotency invariant holds (repo exists  <=>  the run finished). A failed run
    # leaves no repo and is retried on the next session.
    (local_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if status == "ok":
        _upload_results(repo, local_dir / "results.json", token)
    else:
        print(f"run status={status} — results.json kept locally only ({local_dir}); "
              f"no repo created, so this run will be retried.")

    _print_readout(results, repo)
    return results


def check_completion(conditions, seeds, out_root="/kaggle/working/runs"):
    """Roll-call: which of the expected per-run repos exist on the Hub."""
    from huggingface_hub import repo_exists, whoami

    token = _hf_token()
    if not token:
        raise RuntimeError("An HF token is REQUIRED to check completion on the Hub. Set HF_TOKEN.")
    user = whoami(token=token)["name"]
    expected = [repo_id(r, d, s, user) for (r, d) in conditions for s in seeds]
    done = [repo for repo in expected if repo_exists(repo, token=token)]
    missing = missing_runs(done, expected)
    print(f"completion: {len(done)}/{len(expected)} runs done")
    done_set = set(done)
    for repo in expected:
        print(f"  {'✅' if repo in done_set else '⬜'} {repo}")
    return {"expected": expected, "done": done, "missing": sorted(missing)}


def run_calibration(out_root="/kaggle/working/runs"):
    """Run the first real run (strict/easy/seed0) at full frozen config + print the decision."""
    results = run_cell("strict", "easy", 0, out_root=out_root)

    trend = results["reward_trend"]
    runtime_sec = results.get("runtime_sec", 0.0)
    hours = runtime_sec / 3600.0
    weeks_for_all = (N_TOTAL_RUNS * hours / GPU_HOURS_PER_WEEK) if hours > 0 else 0.0

    print()
    print("CALIBRATION DECISION (pre-committed rule)")
    print(f"  reward trended up?   -> {trend['trended_up']}  "
          f"(first-q {trend['first_q_mean']} → last-q {trend['last_q_mean']})")
    print(f"  NaN/instability?     -> {results['status']}")
    print(f"  runtime workable?    -> {_fmt_hm(runtime_sec)}  (≈ {weeks_for_all:.1f} weeks for all 12)")
    print("  → KEEP as run #1 of 12 if: trended up AND status ok AND runtime workable.")
    print("  → Otherwise: adjust config ONCE, re-freeze, discard this run, re-run all 12 fresh.")
    return results
