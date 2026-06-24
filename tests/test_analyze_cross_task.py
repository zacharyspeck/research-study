"""Tests for the Phase-6 cross-task analysis (laptop, no GPU). Synthetic fixtures keep the suite
self-contained + green; a skip-if-present test also runs the REAL Task A judge_results when local."""

import json
from pathlib import Path

import pytest

from evaluation import analyze_cross_task as AX

FIELDS_B = ("company", "round", "raise", "valuation", "founders")


# --------------------------------------------------------------------------- #
# Fixtures (correct real shapes).                                              #
# --------------------------------------------------------------------------- #
def _a_band(strict):
    loose = min(strict + 70, 100.0)
    return {"n": 100, "strict_pct": float(strict), "format_valid_pct": 100.0, "loose_pct": float(loose),
            "passk_pct": float(min(strict + 5, 100)), "cut_off": 0, "gap_pp": round(loose - strict, 1)}


# strict_pct per (cell, band, seed) — chosen so the reward dial on easy_hard/hard OVERLAPS (null)
# but on easy is disjoint.
A_CELLS = {
    "strict-easy":      {"easy": [78, 80, 82], "hard": [9, 10, 11]},
    "loose-easy":       {"easy": [60, 61, 62], "hard": [8, 10, 12]},
    "strict-easy_hard": {"easy": [69, 70, 71], "hard": [10, 11, 12]},
    "loose-easy_hard":  {"easy": [60, 61, 62], "hard": [11, 13, 15]},
}


def _write_a(d, cells=A_CELLS, baseline=(56.0, 9.0)):
    d = Path(d)
    for cell, bands in cells.items():
        rm, df = cell.split("-", 1)
        for seed in (0, 1, 2):
            run = {"reward_mode": rm, "difficulty": df, "seed": seed, "is_baseline": False,
                   "ood_set_sha256": "x", "judge_config_sha256": "y",
                   "metrics": {b: _a_band(bands[b][seed]) for b in ("easy", "hard")}}
            sub = d / f"{cell}-seed{seed}"; sub.mkdir(parents=True)
            (sub / "results.json").write_text(json.dumps(run), encoding="utf-8")
    sub = d / "baseline-untrained-seed-1"; sub.mkdir(parents=True)
    (sub / "results.json").write_text(json.dumps(
        {"reward_mode": "baseline", "difficulty": "untrained", "seed": -1, "is_baseline": True,
         "ood_set_sha256": "x", "judge_config_sha256": "y",
         "metrics": {"easy": _a_band(baseline[0]), "hard": _a_band(baseline[1])}}), encoding="utf-8")
    return str(d)


def _b_band(all5, sm, fstrict, fgap):
    return {"n": 100, "all5_pct": float(all5), "strict_mean_pct": float(sm), "format_valid_pct": 100.0,
            "passk_all5_pct": float(min(all5 + 5, 100)), "passk_field_pct": float(min(sm + 5, 100)),
            "cut_off": 0, "field_strict_pct": {f: float(fstrict) for f in FIELDS_B},
            "field_loose_pct": {f: float(min(fstrict + fgap, 100)) for f in FIELDS_B},
            "field_gap_pp": {f: float(fgap) for f in FIELDS_B}}


def _write_b(d):
    d = Path(d)
    for cell in ("strict-easy", "loose-easy", "strict-easy_hard", "loose-easy_hard"):
        rm, df = cell.split("-", 1)
        for seed in (0, 1, 2):
            o = seed - 1
            run = {"reward_mode": rm, "difficulty": df, "seed": seed, "is_baseline": False, "task": "B",
                   "ood_set_b_sha256": "x", "judge_config_b_sha256": "y",
                   "metrics": {"easy": _b_band(70 + o, 95 + o, 95 + o, 4.0),
                               "hard": _b_band(12 + o, 63 + o, 63 + o, 25.0)}}
            sub = d / f"{cell}-seed{seed}"; sub.mkdir(parents=True)
            (sub / "results.json").write_text(json.dumps(run), encoding="utf-8")
    sub = d / "baseline-untrained-seed-1"; sub.mkdir(parents=True)
    (sub / "results.json").write_text(json.dumps(
        {"reward_mode": "baseline", "difficulty": "untrained", "seed": -1, "is_baseline": True, "task": "B",
         "ood_set_b_sha256": "x", "judge_config_b_sha256": "y",
         "metrics": {"easy": _b_band(60, 90, 90, 8), "hard": _b_band(10, 55, 55, 30)}}), encoding="utf-8")
    return str(d)


# --------------------------------------------------------------------------- #
# Ingest + per-task table (no blended band).                                  #
# --------------------------------------------------------------------------- #
def test_load_taskA_and_no_blended_band(tmp_path):
    res = AX.load_results(_write_a(tmp_path / "A"))
    assert res["task"] == "A" and set(res["cells"]) == set(AX._all_cells()) and res["baseline"] is not None
    for cell in AX._all_cells():
        assert sorted(res["cells"][cell]) == [0, 1, 2]
    cm = AX.cell_metric(res, "strict-easy", "easy", res["spec"]["headline"])
    assert cm["n"] == 3 and cm["mean"] == pytest.approx(80.0) and cm["std"] == pytest.approx(2.0)
    table = AX.render_per_task_table(res)
    assert "strict exact-match" in table and "overall" not in table.lower()   # never a blended band


# --------------------------------------------------------------------------- #
# dial_effect asserts EXACTLY one dial; overlap is a NULL, never a winner.     #
# --------------------------------------------------------------------------- #
def test_dial_effect_exactly_one_dial(tmp_path):
    res = AX.load_results(_write_a(tmp_path / "A"))
    assert AX.reward_dial_effect(res, "easy_hard")["dial"] == "reward_mode"
    assert AX.task_dial_effect(res, "strict")["dial"] == "difficulty"
    with pytest.raises(ValueError):                       # two dials differ -> no ad-hoc subtraction
        AX.dial_effect(res, "strict-easy", "loose-easy_hard", res["spec"]["headline"])
    with pytest.raises(ValueError):                       # zero dials differ
        AX.dial_effect(res, "strict-easy", "strict-easy", res["spec"]["headline"])


def test_overlap_is_null_not_winner(tmp_path):
    res = AX.load_results(_write_a(tmp_path / "A"))
    rd = AX.reward_dial_effect(res, "easy_hard")
    assert rd["bands"]["hard"]["overlap"] is True         # 11±1 vs 13±2 overlap -> NULL
    assert rd["bands"]["easy"]["overlap"] is False        # 70±1 vs 61±1 disjoint
    bands = set(rd["bands"].keys())
    assert bands == {"easy", "hard"}                       # never a blended band
    line = AX._delta_line(rd)
    assert "no effect detected" in line
    assert "beats" not in line.lower() and "win" not in line.lower()
    assert AX.intervals_overlap((10, 12), (11, 15)) and not AX.intervals_overlap((69, 71), (60, 62))


# --------------------------------------------------------------------------- #
# Guards: missing seed, 0-1 scale, baseline separate.                         #
# --------------------------------------------------------------------------- #
def test_missing_seed_flag(tmp_path):
    res = AX.load_results(_write_a(tmp_path / "A"))
    del res["cells"]["strict-easy"][2]                     # now n=2
    cm = AX.cell_metric(res, "strict-easy", "hard", "strict_pct")
    assert cm["n"] == 2 and cm["missing_seed"] is True and "n=2!" in AX._ms(cm)


def test_unit_normalization_catches_0_1_scale(tmp_path):
    bad = {"reward_mode": "strict", "difficulty": "easy", "seed": 0, "is_baseline": False,
           "metrics": {"easy": {"n": 100, "strict_pct": 0.8, "format_valid_pct": 1.0, "loose_pct": 0.99,
                                "passk_pct": 0.85, "cut_off": 0, "gap_pp": 0.19},
                       "hard": {"n": 100, "strict_pct": 0.1, "format_valid_pct": 1.0, "loose_pct": 0.8,
                                "passk_pct": 0.18, "cut_off": 0, "gap_pp": 0.7}}}
    p = tmp_path / "strict-easy-seed0"; p.mkdir()
    (p / "results.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="0-1 scale"):
        AX.load_run(p / "results.json")


def test_baseline_kept_separate(tmp_path):
    res = AX.load_results(_write_a(tmp_path / "A"))
    bvt = AX.baseline_vs_trained(res)
    assert "did training help" in bvt["label"] and "NOT a dial effect" in bvt["label"]
    # the baseline run is routed to res['baseline'], never into a dial-effect cell
    assert res["baseline"]["is_baseline"] is True
    assert all(res["baseline"] is not r for cell in AX._all_cells() for r in res["cells"][cell].values())
    assert bvt["cells"]["strict-easy"]["easy"]["baseline"] == pytest.approx(56.0)
    assert bvt["cells"]["strict-easy"]["easy"]["delta"] == pytest.approx(80.0 - 56.0)


def test_load_fails_loud_on_bad_input(tmp_path):
    def _w(name, run):
        p = tmp_path / name
        p.mkdir(parents=True)
        (p / "results.json").write_text(json.dumps(run), encoding="utf-8")
        return p / "results.json"

    base = {"reward_mode": "strict", "difficulty": "easy", "seed": 0, "is_baseline": False,
            "metrics": {"easy": _a_band(80), "hard": _a_band(10)}}
    with pytest.raises(KeyError):                                   # missing a REQUIRED_TOP key
        AX.load_run(_w("c1", {k: v for k, v in base.items() if k != "is_baseline"}))
    with pytest.raises(KeyError):                                   # missing a band
        AX.load_run(_w("c2", {**base, "metrics": {"easy": _a_band(80)}}))
    with pytest.raises(KeyError):                                   # undetectable task
        AX.load_run(_w("c3", {**base, "metrics": {"easy": {"n": 100, "format_valid_pct": 100.0},
                                                  "hard": {"n": 100, "format_valid_pct": 100.0}}}))


def test_analyze_is_read_only_and_writes_to_target(tmp_path):
    src = Path(_write_a(tmp_path / "A"))
    before = {p: p.read_bytes() for p in sorted(src.rglob("*")) if p.is_file()}
    out = tmp_path / "report.txt"
    report = AX.analyze(str(src), to=str(out))
    assert out.exists() and out.read_text(encoding="utf-8") == report and "PHASE-6" in report
    after = {p: p.read_bytes() for p in sorted(src.rglob("*")) if p.is_file()}
    assert before == after          # input dir byte-identical (read-only); nothing added or modified


def test_gap_concentration_hard_on_top(tmp_path):
    res = AX.load_results(_write_a(tmp_path / "A"))
    rows = AX.gap_concentration(res)
    assert rows[0]["band"] == "hard" and rows[0]["field"] is None   # Task A scalar gap, hard biggest


# --------------------------------------------------------------------------- #
# Task B + cross-task: compares HEADLINES (A strict <-> B all-5-exact), never  #
# A strict <-> B per-field mean.                                               #
# --------------------------------------------------------------------------- #
def test_taskB_perfield_gap_and_cross_task_uses_all5(tmp_path):
    res_a = AX.load_results(_write_a(tmp_path / "A"))
    res_b = AX.load_results(_write_b(tmp_path / "B"))
    assert res_b["task"] == "B" and res_b["spec"]["headline"] == "all5_pct"
    assert res_b["fields"] == sorted(FIELDS_B)
    assert AX.gap_concentration(res_b)[0]["field"] is not None      # Task B gap is PER FIELD
    ct = AX.cross_task_section(res_a, res_b)
    assert ct["headline_A"] == "strict_pct" and ct["headline_B"] == "all5_pct"
    for comp in ct["comparisons"]:
        assert comp["A"]["metric"] == "strict_pct"
        assert comp["B"]["metric"] == "all5_pct"                    # NOT the per-field mean
        assert comp["B"]["metric"] != "strict_mean_pct"
    # Task-B-pending message when only A is present
    assert "Task B pending" in AX.render_cross_task(res_a, None)


# --------------------------------------------------------------------------- #
# Real Task A judge_results (skips on a fresh clone where they're absent).     #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not Path("judge_results").exists(), reason="real Task A judge_results not present")
def test_real_taskA_smoke():
    res = AX.load_results("judge_results")
    assert res["task"] == "A" and len(res["cells"]) == 4 and res["baseline"] is not None
    assert AX.cell_metric(res, "strict-easy_hard", "hard", "strict_pct")["mean"] == pytest.approx(11.0)
    assert AX.reward_dial_effect(res, "easy_hard")["bands"]["hard"]["overlap"] is True   # null at n=3
