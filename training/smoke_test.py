"""Phase-3 Part 2 smoke test — a throwaway dress rehearsal for the pipeline.

Run from a Kaggle cell:  from training.smoke_test import run_smoke; run_smoke()

Proves, on a tiny throwaway run, that GRPO+LoRA trains, that an adapter can be
saved out of the session and reloaded (locally and from the Hub) and that the
reloaded adapter actually changes behavior — BEFORE spending GPU on the 12 runs.
It never touches the sealed OOD set or the baseline/ folder.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import study_config as cfg

THROWAWAY_SEED = 99991  # deliberately not 0/1/2 (those are for the real runs)
PROBE_PROMPT = "A startup raised $3M at a $12M pre-money valuation."


def run_smoke(out_dir="/kaggle/working/smoke"):
    import torch

    from training.train_grpo import prove_adapter_loaded, train_one

    out_dir = Path(out_dir)
    # Guardrail: never the baseline folder, never the OOD set.
    assert "baseline" not in str(out_dir).lower(), "smoke must NOT write to the baseline/ folder"
    adapter_dir = out_dir / "adapter"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Phase-3 Part 2 SMOKE TEST (throwaway dress rehearsal)")
    print("=" * 72)
    smoke = dict(cfg.SMOKE)
    print(f"throwaway seed = {THROWAWAY_SEED}")
    print(f"scratch dir    = {out_dir}")
    print(f"SMOKE overrides= {smoke}")

    # Decide the off-session push target up front (one training run does both).
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    save_repo = None
    if token:
        from huggingface_hub import HfApi
        whoami = HfApi().whoami(token=token)["name"]
        save_repo = f"{whoami}/rlvr-smoke-test"
        print(f"HF token found — will push throwaway adapter to private repo {save_repo}")
    else:
        print("No HF_TOKEN — Hub push will be SKIPPED (see note in the summary).")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 2. Tiny strict / easy run; print the reward trend so we can SEE it move.
    result = train_one("strict", "easy", THROWAWAY_SEED, adapter_dir, save_repo=save_repo, **smoke)
    trend = [(h["step"], round(h["reward"], 4)) for h in result["reward_history"]]
    print("\nreward trend (step, mean):", trend)
    reward_moved = bool(result["reward_varied"])

    # 3. Reload from the LOCAL path; the proof is that the LoRA weights loaded
    #    (present + non-zero). base-vs-trained text is shown for information only.
    print("\n--- proving adapter reload from LOCAL path ---")
    local_proof = prove_adapter_loaded(cfg.MODEL_NAME, str(adapter_dir), PROBE_PROMPT)
    reload_loaded = bool(local_proof["loaded"])
    reload_text_differs = bool(local_proof["differ"])

    # 4. Off-session path: reload from the HUB (the real path the 12 runs need).
    offsession = "skipped-needs-token"
    if save_repo and result["adapter_repo"]:
        print(f"\n--- proving adapter reload from the HUB ({save_repo}) ---")
        hub_proof = prove_adapter_loaded(cfg.MODEL_NAME, save_repo, PROBE_PROMPT)
        offsession = "ok" if hub_proof["loaded"] else "FAILED"
    else:
        print("\n⚠️  HF push SKIPPED — no HF_TOKEN. A token is REQUIRED before the 12 real runs: "
              "Kaggle wipes local files when the session ends, so each adapter must be pushed to "
              "the Hub to survive off-session.")

    # 5. Peak GPU memory.
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"\npeak GPU memory: {peak_gb:.2f} GB")
    else:
        print("\npeak GPU memory: n/a (no CUDA on this machine)")

    # 6. PASS / FAIL summary.
    c_label = {"ok": "PASS", "skipped-needs-token": "SKIP (token needed before 12 runs)",
               "FAILED": "FAIL"}.get(offsession, offsession)
    overall = "PASS" if (reward_moved and reload_loaded
                         and offsession in ("ok", "skipped-needs-token")) else "FAIL"
    print("\n" + "=" * 72)
    print("SMOKE SUMMARY")
    print(f"  (a) reward moved across steps:           {'PASS' if reward_moved else 'FAIL'}")
    print(f"  (b) reloaded adapter loaded (LoRA != 0): {'PASS' if reload_loaded else 'FAIL'}"
          f"   [text differs from base: {reload_text_differs}]")
    print(f"  (c) off-session (Hub) push + reload:     {c_label}")
    print(f"  OVERALL: {overall}")
    print("=" * 72)

    return {
        "reward_moved": reward_moved,
        "reload_loaded": reload_loaded,
        "reload_text_differs": reload_text_differs,
        "offsession": offsession,
        "overall": overall,
        "adapter_path": result["adapter_path"],
        "adapter_repo": result["adapter_repo"],
    }
