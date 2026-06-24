"""Phase-6 cross-task analysis — measures and COMPARES; it does NOT conclude.

Laptop-only, READ-ONLY convenience tooling over the frozen judge_results: runs no model, touches
no GPU, modifies no training/judging/grader code. It ingests Task A and/or Task B judge results and
emits tables, deltas, overlap (null) flags, and reward-hacking gap concentration — and NOTHING else.

NON-NEGOTIABLES (Principles 1, 8, 10):
  * It never bakes in or asserts "the task dial wins" (the pre-registered hypothesis) or any verdict.
  * It reports a NULL (overlapping mean±std intervals -> no detected effect) as cleanly as a real
    effect. dial deltas carry a sign + magnitude + an OVERLAP flag; the script NEVER says "X beats Y".
  * Bands are never blended; no code path returns a band-blended comparison number.
  * Task A headline = strict exact-match; Task B headline = ALL-5-EXACT, with per-field mean reported
    SEPARATELY and labeled "diagnostic". Cross-task compares HEADLINE <-> HEADLINE only (A strict <->
    B all-5-exact), NEVER A strict <-> B per-field mean.

Spread definition (stated once, used identically for both tasks): SAMPLE standard deviation
(statistics.stdev, ddof=1) over the seeds; 0.0 for n<2. n=3 in the full design.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REWARD_MODES = ("strict", "loose")
DIFFICULTIES = ("easy", "easy_hard")
BANDS = ("easy", "hard")
SEEDS = (0, 1, 2)
COUNT_KEYS = ("n", "cut_off")           # not percent-scale; excluded from the range/scale checks

# Per-task metric schema. headline is the record-correctness number; per-field mean is a SEPARATE
# diagnostic for Task B and must never be substituted for the headline.
TASK_SPECS = {
    "A": {
        "name": "A",
        "headline": "strict_pct",
        "headline_label": "strict exact-match",
        "diagnostics": ("format_valid_pct", "loose_pct", "passk_pct"),
        "gap": {"kind": "scalar", "key": "gap_pp"},
        "hash_keys": ("ood_set_sha256", "judge_config_sha256"),
    },
    "B": {
        "name": "B",
        "headline": "all5_pct",
        "headline_label": "all-5-exact (record correctness)",
        "diagnostics": ("strict_mean_pct", "format_valid_pct", "passk_all5_pct", "passk_field_pct"),
        "gap": {"kind": "per_field", "key": "field_gap_pp"},
        "hash_keys": ("ood_set_b_sha256", "judge_config_b_sha256"),
    },
}
REQUIRED_TOP = ("reward_mode", "difficulty", "seed", "is_baseline", "metrics")


def _all_cells():
    return [f"{rm}-{df}" for rm in REWARD_MODES for df in DIFFICULTIES]


# --------------------------------------------------------------------------- #
# Ingest (READ-ONLY) — detect task, assert keys, normalize to percent.         #
# --------------------------------------------------------------------------- #
def _percent_values(run):
    """Every percent-scale metric value in a run (flattens Task-B field dicts; excludes counts)."""
    vals = []
    for m in run["metrics"].values():
        for k, v in m.items():
            if k in COUNT_KEYS:
                continue
            if isinstance(v, dict):
                vals.extend(float(x) for x in v.values())
            elif v is not None:
                vals.append(float(v))
    return vals


def _range_check(run, path):
    """Normalize/guard: every percent metric must be in [0, 100]; catch a 0-1-vs-percent mismatch
    before it propagates (real data has format-validity ~100, so max <= 1.0 means a 0-1 scale)."""
    vals = _percent_values(run)
    for v in vals:
        if not (-0.05 <= v <= 100.05):
            raise ValueError(f"{path}: metric value {v} outside [0,100] — wrong scale or corrupt input.")
    if vals and 0 < max(vals) <= 1.0:
        raise ValueError(f"{path}: all percent metrics <= 1.0 (max={max(vals)}) — looks like a 0-1 "
                         f"scale, expected 0-100 percent. Refusing to misread.")


def detect_task(metrics_band) -> str:
    if "all5_pct" in metrics_band:
        return "B"
    if "strict_pct" in metrics_band:
        return "A"
    raise KeyError("cannot detect task: neither 'all5_pct' (B) nor 'strict_pct' (A) in metrics band")


def load_run(path) -> dict:
    """Load + validate one results.json. Fails loud on a missing/renamed key (never silently misread)."""
    run = json.loads(Path(path).read_text(encoding="utf-8"))
    for k in REQUIRED_TOP:
        if k not in run:
            raise KeyError(f"{path}: missing required key '{k}' (renamed/old format?)")
    for band in BANDS:
        if band not in run["metrics"]:
            raise KeyError(f"{path}: metrics missing band '{band}'")
    task = detect_task(run["metrics"]["hard"])
    spec = TASK_SPECS[task]
    for mk in (spec["headline"], *spec["diagnostics"]):
        for band in BANDS:
            if mk not in run["metrics"][band]:
                raise KeyError(f"{path}: task {task} band '{band}' missing metric '{mk}'")
    _range_check(run, path)
    run["_task"] = task
    return run


def load_results(out_dir) -> dict:
    """Load all */results.json in a judge-output dir (READ-ONLY). One task per dir."""
    out_dir = Path(out_dir)
    files = sorted(out_dir.glob("*/results.json"))
    if not files:
        raise FileNotFoundError(f"no */results.json under {out_dir}")
    runs = [load_run(p) for p in files]
    tasks = {r["_task"] for r in runs}
    if len(tasks) != 1:
        raise ValueError(f"{out_dir}: mixed tasks {tasks} — expected exactly one task per results dir.")
    task = tasks.pop()
    cells, baseline = {}, None
    for r in runs:
        if r["is_baseline"]:
            baseline = r
        else:
            cells.setdefault(f"{r['reward_mode']}-{r['difficulty']}", {})[int(r["seed"])] = r
    fields = (sorted(runs[0]["metrics"]["hard"]["field_gap_pp"].keys()) if task == "B" else None)
    return {"task": task, "spec": TASK_SPECS[task], "cells": cells, "baseline": baseline,
            "fields": fields, "dir": str(out_dir)}


# --------------------------------------------------------------------------- #
# Aggregation across seeds (mean ± spread). Spread = sample std (ddof=1).      #
# --------------------------------------------------------------------------- #
def mean_spread(values) -> dict:
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None, "lo": None, "hi": None}
    mean = sum(vals) / n
    std = statistics.stdev(vals) if n >= 2 else 0.0
    return {"n": n, "mean": mean, "std": std, "min": min(vals), "max": max(vals),
            "lo": mean - std, "hi": mean + std}


def _get_metric(run, band, metric):
    m = run["metrics"][band]
    if isinstance(metric, tuple):          # ("field_strict_pct", "company")
        return m[metric[0]][metric[1]]
    return m[metric]


def cell_metric(results, cell, band, metric) -> dict:
    """Aggregate one metric for one (cell, band) across its seeds. Flags a missing seed loudly."""
    seed_runs = results["cells"].get(cell, {})
    vals = [_get_metric(seed_runs[s], band, metric) for s in SEEDS if s in seed_runs]
    ms = mean_spread(vals)
    ms["seeds_present"] = sorted(seed_runs.keys())
    # Flag from the COUNT ACTUALLY AGGREGATED (n over canonical seeds), not the raw key count, so an
    # off-canonical / duplicate seed key can never mask a partial mean as if it were n=3.
    ms["missing_seed"] = ms["n"] != len(SEEDS)
    return ms


# --------------------------------------------------------------------------- #
# Dial effects — ONE function, asserts EXACTLY ONE dial differs. No winner.    #
# --------------------------------------------------------------------------- #
def _parse_cell(cell):
    rm, df = cell.split("-", 1)
    if rm not in REWARD_MODES or df not in DIFFICULTIES:
        raise ValueError(f"unrecognized cell {cell!r}")
    return rm, df


def _one_dial(cell_a, cell_b) -> str:
    a, b = _parse_cell(cell_a), _parse_cell(cell_b)
    names = ("reward_mode", "difficulty")
    diff = [names[i] for i in (0, 1) if a[i] != b[i]]
    if len(diff) != 1:
        raise ValueError(f"dial_effect requires cells differing in EXACTLY ONE dial; "
                         f"{cell_a} vs {cell_b} differ in {len(diff)}: {diff or 'none'}")
    return diff[0]


def intervals_overlap(a, b) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def dial_effect(results, cell_a, cell_b, metric) -> dict:
    """Delta of `metric` between two cells that differ in EXACTLY ONE dial (raises otherwise).
    Per band: the signed delta, the mean±std for each cell, and an OVERLAP (null) flag. It reports
    the delta; it NEVER labels a winner. Overlapping intervals -> 'no effect detected (n=3, small)'."""
    dial = _one_dial(cell_a, cell_b)
    out = {"cell_a": cell_a, "cell_b": cell_b, "dial": dial, "metric": metric, "bands": {}}
    for band in BANDS:
        ma, mb = cell_metric(results, cell_a, band, metric), cell_metric(results, cell_b, band, metric)
        if ma["n"] == 0 or mb["n"] == 0:
            out["bands"][band] = {"available": False, "a": ma, "b": mb}
            continue
        overlap = intervals_overlap((ma["lo"], ma["hi"]), (mb["lo"], mb["hi"]))
        out["bands"][band] = {
            "available": True, "a": ma, "b": mb,
            "delta": mb["mean"] - ma["mean"],
            "overlap": overlap,                 # True -> NULL: no effect detected (n=3, small)
            "missing_seed": ma["missing_seed"] or mb["missing_seed"],
        }
    return out


def reward_dial_effect(results, difficulty, metric=None) -> dict:
    """Reward dial: strict -> loose, task/difficulty FIXED. Defaults to the task's headline."""
    metric = metric or results["spec"]["headline"]
    return dial_effect(results, f"strict-{difficulty}", f"loose-{difficulty}", metric)


def task_dial_effect(results, reward, metric=None) -> dict:
    """Task dial: easy -> easy_hard, reward FIXED. Defaults to the task's headline."""
    metric = metric or results["spec"]["headline"]
    return dial_effect(results, f"{reward}-easy", f"{reward}-easy_hard", metric)


# --------------------------------------------------------------------------- #
# Baseline (trained vs untrained) — SEPARATE from dial effects, labeled.       #
# --------------------------------------------------------------------------- #
def baseline_vs_trained(results, metric=None) -> dict | None:
    """'Did training help' = trained cell mean vs the untrained baseline (n=1). This is a SEPARATE
    output from the trained-vs-trained dial effects and is never conflated with them."""
    if results["baseline"] is None:
        return None
    metric = metric or results["spec"]["headline"]
    out = {"metric": metric, "label": "trained vs UNTRAINED baseline (did training help) — NOT a dial effect",
           "cells": {}}
    for cell in _all_cells():
        out["cells"][cell] = {}
        for band in BANDS:
            tr = cell_metric(results, cell, band, metric)
            bv = _get_metric(results["baseline"], band, metric)
            out["cells"][cell][band] = {
                "baseline": bv, "trained_mean": tr["mean"], "n_seeds": tr["n"],
                "delta": (tr["mean"] - bv) if tr["mean"] is not None else None,
                "missing_seed": tr["missing_seed"]}
    return out


# --------------------------------------------------------------------------- #
# Reward-hacking gap concentration — magnitudes only, no interpretation.       #
# --------------------------------------------------------------------------- #
def gap_concentration(results) -> list:
    """Where the loose-strict gap is largest: per cell/band (Task A scalar) or per FIELD per band
    (Task B). Returns rows sorted by gap magnitude. Reports magnitudes; draws no conclusion."""
    spec = results["spec"]
    rows = []
    for cell in _all_cells():
        for band in BANDS:
            if spec["gap"]["kind"] == "scalar":
                cm = cell_metric(results, cell, band, spec["gap"]["key"])
                if cm["n"]:
                    rows.append({"cell": cell, "band": band, "field": None,
                                 "gap_mean": cm["mean"], "gap_std": cm["std"], "missing_seed": cm["missing_seed"]})
            else:
                for f in (results["fields"] or []):
                    cm = cell_metric(results, cell, band, (spec["gap"]["key"], f))
                    if cm["n"]:
                        rows.append({"cell": cell, "band": band, "field": f,
                                     "gap_mean": cm["mean"], "gap_std": cm["std"], "missing_seed": cm["missing_seed"]})
    rows.sort(key=lambda r: (r["gap_mean"] if r["gap_mean"] is not None else -1.0), reverse=True)
    return rows


# --------------------------------------------------------------------------- #
# Cross-task (both present) — side by side, HEADLINE<->HEADLINE, no verdict.   #
# --------------------------------------------------------------------------- #
def cross_task_section(res_a, res_b) -> dict:
    """Place the two tasks' HEADLINE dial effects side by side so a human can see whether the dials
    behaved the same way. Compares LIKE metrics only — A strict exact-match <-> B all-5-exact (each
    task's own headline), NEVER A strict <-> B per-field mean. Draws NO agreement conclusion."""
    hl_a, hl_b = res_a["spec"]["headline"], res_b["spec"]["headline"]
    comparisons = []
    for label, kind, fixed in [
        ("reward dial (strict->loose, difficulty=easy_hard)", "reward", "easy_hard"),
        ("task dial (easy->easy_hard, reward=strict)", "task", "strict"),
    ]:
        if kind == "reward":
            a, b = reward_dial_effect(res_a, fixed, hl_a), reward_dial_effect(res_b, fixed, hl_b)
        else:
            a, b = task_dial_effect(res_a, fixed, hl_a), task_dial_effect(res_b, fixed, hl_b)
        comparisons.append({"dial": label, "A": a, "B": b})
    return {"headline_A": hl_a, "headline_B": hl_b,
            "note": "side-by-side; each task uses its OWN headline (A: strict exact-match, B: all-5-exact). "
                    "One slice each (reward dial at difficulty=easy_hard; task dial at reward=strict) — see "
                    "the per-task DIAL EFFECTS for all slices. No agreement conclusion is drawn.",
            "comparisons": comparisons}


# --------------------------------------------------------------------------- #
# Rendering (text only).                                                       #
# --------------------------------------------------------------------------- #
def _ms(m):
    if m["n"] == 0:
        return "--"
    flag = f"(n={m['n']}!)" if m["n"] != len(SEEDS) else ""   # any partial mean is flagged, always
    return f"{m['mean']:.1f}±{m['std']:.1f}{flag}"


def _delta_line(eff):
    L = []
    for band in BANDS:
        d = eff["bands"][band]
        if not d["available"]:
            L.append(f"    {band:<5}: unavailable (a n={d['a']['n']}, b n={d['b']['n']})")
            continue
        # Both tags are descriptive facts + an explicit no-verdict reminder; neither names a winner.
        tag = ("  NULL: intervals overlap — no effect detected (n=3, small)" if d["overlap"]
               else "  intervals disjoint — signed delta only, not a verdict")
        miss = "  MISSING-SEED!" if d["missing_seed"] else ""
        L.append(f"    {band:<5}: Δ={d['delta']:+.1f}pp   [{eff['cell_a']} {_ms(d['a'])}  vs  "
                 f"{eff['cell_b']} {_ms(d['b'])}]{tag}{miss}")
    return "\n".join(L)


def render_per_task_table(results) -> str:
    spec, task = results["spec"], results["task"]
    L = ["", "=" * 96,
         f"PER-TASK TABLE — Task {task}  (HEADLINE = {spec['headline_label']}; mean ± std over 3 seeds)",
         "=" * 96]
    cols = (spec["headline"], *spec["diagnostics"])
    L.append(f"{'cell':<18}{'band':<6}" + "".join(f"{c:>18}" for c in cols)
             + ("   [Task B: strict_mean_pct is the per-field-mean DIAGNOSTIC, not the headline]" if task == "B" else ""))
    for cell in _all_cells():
        for band in BANDS:
            L.append(f"{cell:<18}{band:<6}"
                     + "".join(f"{_ms(cell_metric(results, cell, band, c)):>18}" for c in cols))
    if results["fields"]:
        L.append("\nper-field strict (Task B DIAGNOSTIC — never the headline):")
        for cell in _all_cells():
            for band in BANDS:
                vals = "  ".join(f"{f[:3]}={_ms(cell_metric(results, cell, band, ('field_strict_pct', f)))}"
                                 for f in results["fields"])
                L.append(f"  {cell:<18}{band:<6}{vals}")
    return "\n".join(L)


def render_dial_effects(results) -> str:
    spec = results["spec"]
    L = ["", "-" * 96, f"DIAL EFFECTS — Task {results['task']} (headline = {spec['headline']}); "
         f"deltas + null flags only, NO winner.", "-" * 96]
    L.append("REWARD dial (strict->loose), task fixed — per band below (deltas + null flags; NO verdict):")
    for df in DIFFICULTIES:
        L.append(f"  difficulty={df}:")
        L.append(_delta_line(reward_dial_effect(results, df)))
    L.append("TASK dial (easy->easy_hard), reward fixed:")
    for rm in REWARD_MODES:
        L.append(f"  reward={rm}:")
        L.append(_delta_line(task_dial_effect(results, rm)))
    return "\n".join(L)


def render_gap(results) -> str:
    rows = gap_concentration(results)
    L = ["", "-" * 96, f"REWARD-HACKING GAP CONCENTRATION — Task {results['task']} (loose-strict, "
         + ("per cell/band" if results["spec"]["gap"]["kind"] == "scalar" else "per FIELD per band")
         + "); magnitudes only.", "-" * 96]
    for r in rows[:12]:
        where = f"{r['cell']}/{r['band']}" + (f"/{r['field']}" if r["field"] else "")
        flag = " (n<3!)" if r["missing_seed"] else ""
        L.append(f"  {where:<34} gap = {r['gap_mean']:.1f}±{r['gap_std']:.1f} pp{flag}")
    return "\n".join(L)


def render_baseline(results) -> str:
    bvt = baseline_vs_trained(results)
    if bvt is None:
        return "\n(no baseline present — trained-vs-untrained not reported)"
    L = ["", "-" * 96, f"TRAINED vs UNTRAINED — Task {results['task']} ({bvt['label']}); separate from dial effects.",
         "-" * 96]
    for cell in _all_cells():
        parts = []
        for band in BANDS:
            c = bvt["cells"][cell][band]
            if c["delta"] is None:
                parts.append(f"{band}: --")
            else:
                miss = "(n<3!)" if c["missing_seed"] else ""
                parts.append(f"{band}: {c['baseline']:.1f}->{c['trained_mean']:.1f} (Δ{c['delta']:+.1f}){miss}")
        L.append(f"  {cell:<18}" + "   ".join(parts))
    return "\n".join(L)


def render_cross_task(res_a, res_b) -> str:
    if res_b is None:
        return ("\n" + "=" * 96 + "\nCROSS-TASK: Task B pending — the cross-task comparison activates "
                "only when BOTH tasks' results are present.\n" + "=" * 96)
    ct = cross_task_section(res_a, res_b)
    L = ["", "=" * 96, "CROSS-TASK (side by side) — " + ct["note"],
         f"Task A headline = {ct['headline_A']}   |   Task B headline = {ct['headline_B']}", "=" * 96]
    for comp in ct["comparisons"]:
        L.append(f"\n[{comp['dial']}]")
        for t, eff in (("A", comp["A"]), ("B", comp["B"])):
            L.append(f"  Task {t} ({eff['metric']}):")
            L.append(_delta_line(eff))
    return "\n".join(L)


def analyze(out_dir_a, out_dir_b=None, to=None) -> str:
    """Full report for Task A (and Task B if present). READ-ONLY; writes only to `to` (separate file)."""
    res_a = load_results(out_dir_a)
    res_b = load_results(out_dir_b) if out_dir_b else None
    parts = [f"PHASE-6 CROSS-TASK ANALYSIS  (measures + compares; draws NO verdict; nulls reported as "
             f"cleanly as effects; spread = sample std ddof=1, n=3)",
             render_per_task_table(res_a), render_dial_effects(res_a), render_gap(res_a), render_baseline(res_a)]
    if res_b is not None:
        parts += [render_per_task_table(res_b), render_dial_effects(res_b), render_gap(res_b),
                  render_baseline(res_b)]
    parts.append(render_cross_task(res_a, res_b))
    report = "\n".join(parts)
    if to:
        Path(to).write_text(report, encoding="utf-8")     # separate file; judge_results untouched
    return report


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "judge_results"
    b = sys.argv[2] if len(sys.argv) > 2 else None
    print(analyze(a, b))
