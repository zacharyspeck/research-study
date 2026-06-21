"""Phase-3 untrained baseline runner for Task A (deal-math ownership %).

The GPU run happens on Kaggle: the notebook loads the frozen model ONCE and
calls :func:`run_baseline(model, tokenizer)`. This module never loads the model
itself, and every torch import is deferred into the functions that need it, so
the pure helpers below are unit-testable on a machine with no GPU and no torch.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import study_config as cfg
from data_generation.generate import build_all
from graders.grader import extract_answer, format_valid, grade, grade_loose

# --------------------------------------------------------------------------- #
# Pure, GPU-free logic (this is what tests/test_eval.py exercises).            #
# --------------------------------------------------------------------------- #

ANSWER_IS_RE = re.compile(r"answer\s+is", re.IGNORECASE)


class GuardrailError(RuntimeError):
    """Raised to hard-stop the baseline when a safety precondition fails."""


def check_guardrails(ood_items, train_items, model_name,
                     n_ood_easy=cfg.N_OOD_EASY, n_ood_hard=cfg.N_OOD_HARD):
    """Validate the run before any GPU time is spent.

    Hard-stops (raise :class:`GuardrailError` with a plain-English message):
      (a) the OOD set must have exactly ``n_ood_easy`` easy + ``n_ood_hard`` hard;
      (b) OOD prompts and train prompts must be disjoint (the test set is sealed).
    Soft check (returned as a warning string, never fatal):
      (c) the loaded model name should contain "1.5B".

    Returns the list of non-fatal warnings (empty if all clear).
    """
    n_easy = sum(1 for it in ood_items if it["difficulty"] == "easy")
    n_hard = sum(1 for it in ood_items if it["difficulty"] == "hard")
    if (n_easy, n_hard) != (n_ood_easy, n_ood_hard):
        raise GuardrailError(
            f"OOD test set must contain exactly {n_ood_easy} easy + {n_ood_hard} hard items, "
            f"but it has {n_easy} easy + {n_hard} hard. Refusing to run on the wrong set."
        )

    overlap = {it["prompt"] for it in ood_items} & {it["prompt"] for it in train_items}
    if overlap:
        raise GuardrailError(
            f"{len(overlap)} OOD prompt(s) also appear in the training set, so the OOD set is "
            f"NOT held out. Refusing to run: a leaked test set would invalidate the baseline."
        )

    warnings = []
    if "1.5B" not in (model_name or ""):
        warnings.append(
            f"Loaded model name {model_name!r} does not contain '1.5B'. The frozen config is "
            f"{cfg.MODEL_NAME!r} — double-check you loaded the right model."
        )
    return warnings


def is_cutoff(raw_output, generated_token_count, max_new_tokens=cfg.MAX_NEW_TOKENS):
    """True if generation ran to the token cap without ever stating an answer."""
    hit_cap = generated_token_count >= max_new_tokens
    has_answer = bool(ANSWER_IS_RE.search(raw_output or ""))
    return bool(hit_cap and not has_answer)


def passk_passed(sample_strict_grades):
    """Pass@k: the item passes if ANY of its samples is strict-correct."""
    return any(float(g) == 1.0 for g in sample_strict_grades)


def _pct(numerator, n):
    return round(100.0 * numerator / n, 1) if n else 0.0


def compute_group_metrics(records):
    """Metrics for one list of per-item records."""
    n = len(records)
    return {
        "n": n,
        "strict_pct": _pct(sum(r["strict"] for r in records), n),
        "format_valid_pct": _pct(sum(r["format_valid"] for r in records), n),
        "loose_pct": _pct(sum(r["loose"] for r in records), n),
        "passk_pct": _pct(sum(1 for r in records if r["passk_passed"]), n),
        "cut_off": sum(1 for r in records if r["cut_off"]),
    }


def compute_all_metrics(records):
    """Metrics split by easy / hard / overall."""
    easy = [r for r in records if r["difficulty"] == "easy"]
    hard = [r for r in records if r["difficulty"] == "hard"]
    return {
        "easy": compute_group_metrics(easy),
        "hard": compute_group_metrics(hard),
        "overall": compute_group_metrics(records),
    }


def system_prompt_from_pilot_log(path=None):
    """Read the verbatim system prompt out of the ```text block in PILOT_LOG.md."""
    path = Path(path) if path else _REPO_ROOT / "PILOT_LOG.md"
    text = path.read_text(encoding="utf-8")
    match = re.search(r"```text\n(.*?)\n```", text, re.DOTALL)
    if not match:
        raise ValueError("No ```text system-prompt block found in PILOT_LOG.md")
    return match.group(1).strip()


def git_short_sha():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, cwd=_REPO_ROOT,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def loaded_model_name(model):
    config = getattr(model, "config", None)
    return getattr(config, "_name_or_path", "") or getattr(model, "name_or_path", "") or ""


def _select_sample_transcripts(records, limit=8):
    """Pick a varied, informative subset of greedy transcripts to print."""
    chosen, seen = [], set()

    def take(pred, k):
        count = 0
        for i, r in enumerate(records):
            if i in seen or not pred(r):
                continue
            chosen.append(r)
            seen.add(i)
            count += 1
            if count >= k:
                break

    take(lambda r: r["strict"] == 1.0, 3)                                 # correct
    take(lambda r: r["strict"] == 0.0 and r["extracted"] is not None, 3)  # wrong but numeric
    take(lambda r: r["format_valid"] == 1.0 and r["strict"] == 0.0, 2)    # number produced, still wrong
    take(lambda r: r["extracted"] is None, 2)                             # no number found
    return chosen[:limit]


def _format_table(metrics):
    header = f"{'group':<8}{'n':>5}{'strict':>9}{'format':>9}{'loose':>9}{'pass@k':>9}{'cutoff':>8}"
    lines = [header, "-" * len(header)]
    for grp in ("easy", "hard", "overall"):
        m = metrics[grp]
        lines.append(
            f"{grp:<8}{m['n']:>5}{m['strict_pct']:>8.1f}%{m['format_valid_pct']:>8.1f}%"
            f"{m['loose_pct']:>8.1f}%{m['passk_pct']:>8.1f}%{m['cut_off']:>8}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# GPU path — torch imported lazily inside, never touched by the unit tests.    #
# --------------------------------------------------------------------------- #

def _seed_everything(seed):
    import random as _random

    import torch
    _random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except ImportError:
        pass


def _chat_text(tokenizer, item):
    """system = frozen SYSTEM_PROMPT; user = the item's prompt exactly as-is."""
    messages = [
        {"role": "system", "content": cfg.SYSTEM_PROMPT},
        {"role": "user", "content": item["prompt"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _generate(model, tokenizer, texts, gen_kwargs, batch_size):
    """Batched generation with left padding. Returns [(decoded_text, n_new_tokens)]."""
    import torch
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    results = []
    try:
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, pad_token_id=tokenizer.pad_token_id, **gen_kwargs)
            new = out[:, enc["input_ids"].shape[1]:]
            for row in new:
                n_new = int((row != tokenizer.pad_token_id).sum().item())
                text = tokenizer.decode(row, skip_special_tokens=True).strip()
                results.append((text, n_new))
    finally:
        tokenizer.padding_side = prev_side
    return results


def _write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def _print_transcript(rec, width=320):
    body = rec["raw_output"].replace("\n", " / ")
    if len(body) > width:
        body = body[:width] + "…"
    mark = "✅" if rec["strict"] == 1.0 else "❌"
    print(f"  {mark} [{rec['difficulty']}] gold={rec['gold']} extracted={rec['extracted']} "
          f"strict={rec['strict']:.0f} loose={rec['loose']:.0f} fmt={rec['format_valid']:.0f} "
          f"cutoff={rec['cut_off']}")
    print(f"     output: {body}")


def run_baseline(model, tokenizer, out_dir="/kaggle/working/baseline", batch_size=16):
    """Record the untrained model's 'before' score on the sealed OOD test set.

    The Kaggle notebook passes a model + tokenizer it has ALREADY loaded; this
    function never (re)loads the model.
    """
    import torch  # noqa: F401  — fail loudly here if run on a box without torch

    sha = git_short_sha()
    snap = cfg.snapshot()

    # 1. Print what's running.
    print("=" * 72)
    print("Phase-3 untrained baseline — Task A, sealed OOD test set")
    print("=" * 72)
    print(f"git commit: {sha}")
    print("FROZEN config (study_config.py):")
    for key in sorted(snap):
        value = snap[key]
        if isinstance(value, str) and len(value) > 88:
            value = value[:85] + "..."
        print(f"  {key} = {value!r}")
    print()

    # 2. Build data (frozen seed/sizes); need OOD + train (for the leak check).
    data = build_all(cfg.DATA_SEED, cfg.N_EASY, cfg.N_HARD, cfg.N_OOD_EASY, cfg.N_OOD_HARD)
    ood = data["ood_test"]
    train = data["train_easy"] + data["train_hard"]
    print(f"Built data: {len(ood)} OOD items | {len(train)} train items (for leak check).")

    # 3. Guardrails (hard-stop on failure; warn on model-name mismatch).
    model_name = loaded_model_name(model) or cfg.MODEL_NAME
    for warning in check_guardrails(ood, train, model_name, cfg.N_OOD_EASY, cfg.N_OOD_HARD):
        print("⚠️  WARNING:", warning)
    print("✅ guardrails passed")
    print()

    chat_texts = [_chat_text(tokenizer, it) for it in ood]

    # 4. Greedy pass (correctness / format-validity).
    print(f"Greedy pass over {len(ood)} OOD items ...")
    greedy_records = []
    for it, (text, n_new) in zip(
        ood, _generate(model, tokenizer, chat_texts, cfg.GREEDY_GEN_KWARGS, batch_size)
    ):
        greedy_records.append({
            "prompt": it["prompt"],
            "gold": it["answer"],
            "difficulty": it["difficulty"],
            "raw_output": text,
            "extracted": extract_answer(text),
            "strict": grade(text, it["answer"]),
            "loose": grade_loose(text, it["answer"]),
            "format_valid": format_valid(text),
            "cut_off": is_cutoff(text, n_new, cfg.MAX_NEW_TOKENS),
        })

    # 5. Sampling pass for Pass@k (seeded for reproducibility).
    print(f"Sampling pass: {cfg.PASS_K} samples/item "
          f"(temp={cfg.SAMPLE_TEMPERATURE}, top_p={cfg.SAMPLE_TOP_P}) ...")
    _seed_everything(cfg.EVAL_SEED)
    repeated = [t for t in chat_texts for _ in range(cfg.PASS_K)]
    sample_out = _generate(model, tokenizer, repeated, cfg.SAMPLE_GEN_KWARGS, batch_size)
    passk_records = []
    for idx, it in enumerate(ood):
        group = sample_out[idx * cfg.PASS_K:(idx + 1) * cfg.PASS_K]
        grades = [grade(text, it["answer"]) for text, _ in group]
        passed = passk_passed(grades)
        greedy_records[idx]["passk_passed"] = passed
        passk_records.append({
            "prompt": it["prompt"],
            "gold": it["answer"],
            "difficulty": it["difficulty"],
            "passk_passed": passed,
            "samples": [
                {"raw_output": text, "extracted": extract_answer(text), "strict": g}
                for (text, _), g in zip(group, grades)
            ],
        })

    # 6. Metrics, split easy / hard / overall.
    metrics = compute_all_metrics(greedy_records)

    # 7. Save.
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {
        "commit": sha,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "config": snap,
        "metrics": metrics,
    }
    results_path = out / "baseline_results.json"
    greedy_path = out / "baseline_transcripts_greedy.jsonl"
    passk_path = out / "baseline_transcripts_passk.jsonl"
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    _write_jsonl(greedy_records, greedy_path)
    _write_jsonl(passk_records, passk_path)
    print(f"saved ✅ → {results_path}")
    print(f"saved ✅ → {greedy_path}")
    print(f"saved ✅ → {passk_path}")
    print()

    # 8. Print results table + a varied set of sample transcripts.
    print(_format_table(metrics))
    print()
    print("Sample greedy transcripts (check the grader grabs the right number from messy text):")
    for rec in _select_sample_transcripts(greedy_records):
        _print_transcript(rec)
    print()

    # 9. Return.
    return results
