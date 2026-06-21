"""Pure-Python tests for the baseline evaluation logic (no model, no GPU).

Only the model-free logic is tested here, with hand-made fake model outputs.
Importing evaluation.run_baseline must not import torch (it is deferred), so
these run on a plain machine.
"""

import pytest

from study_config import MAX_NEW_TOKENS, SYSTEM_PROMPT
from evaluation.run_baseline import (
    GuardrailError,
    check_guardrails,
    compute_all_metrics,
    compute_group_metrics,
    is_cutoff,
    passk_passed,
    system_prompt_from_pilot_log,
)


# --------------------------------------------------------------------------- #
# Pass@k aggregation                                                           #
# --------------------------------------------------------------------------- #
def test_passk_passes_if_any_sample_correct():
    assert passk_passed([0.0, 0.0, 1.0, 0.0]) is True
    assert passk_passed([1.0]) is True


def test_passk_fails_if_no_sample_correct():
    assert passk_passed([0.0, 0.0, 0.0, 0.0]) is False
    assert passk_passed([]) is False


# --------------------------------------------------------------------------- #
# Cut-off detection                                                            #
# --------------------------------------------------------------------------- #
def test_cutoff_flags_maxlen_output_with_no_answer():
    assert is_cutoff("a long rambling chain with no conclusion", MAX_NEW_TOKENS, MAX_NEW_TOKENS) is True


def test_cutoff_false_when_answer_pattern_present():
    assert is_cutoff("...therefore The answer is 20", MAX_NEW_TOKENS, MAX_NEW_TOKENS) is False


def test_cutoff_false_when_below_maxlen():
    assert is_cutoff("short reply, no answer here", 15, MAX_NEW_TOKENS) is False


# --------------------------------------------------------------------------- #
# Per-group metric math                                                        #
# --------------------------------------------------------------------------- #
def _rec(strict, loose, fmt, passk, cut, diff):
    return {"strict": strict, "loose": loose, "format_valid": fmt,
            "passk_passed": passk, "cut_off": cut, "difficulty": diff}


def test_group_metrics_one_of_two_correct_is_fifty_percent():
    recs = [
        _rec(1.0, 1.0, 1.0, True, False, "easy"),
        _rec(0.0, 1.0, 1.0, False, True, "easy"),
    ]
    m = compute_group_metrics(recs)
    assert m["n"] == 2
    assert m["strict_pct"] == 50.0
    assert m["loose_pct"] == 100.0
    assert m["format_valid_pct"] == 100.0
    assert m["passk_pct"] == 50.0
    assert m["cut_off"] == 1


def test_group_metrics_empty_is_zero_not_crash():
    m = compute_group_metrics([])
    assert m["n"] == 0
    assert m["strict_pct"] == 0.0
    assert m["cut_off"] == 0


def test_all_metrics_split_by_difficulty():
    recs = [
        _rec(1.0, 1.0, 1.0, True, False, "easy"),
        _rec(0.0, 0.0, 0.0, False, False, "hard"),
    ]
    m = compute_all_metrics(recs)
    assert m["easy"]["n"] == 1 and m["easy"]["strict_pct"] == 100.0
    assert m["hard"]["n"] == 1 and m["hard"]["strict_pct"] == 0.0
    assert m["overall"]["n"] == 2 and m["overall"]["strict_pct"] == 50.0


# --------------------------------------------------------------------------- #
# Guardrails                                                                   #
# --------------------------------------------------------------------------- #
def _items(prompts, difficulty):
    return [{"prompt": p, "difficulty": difficulty, "answer": 12.5} for p in prompts]


def test_guardrails_pass_when_disjoint_and_counts_match():
    ood = _items(["o1"], "easy") + _items(["o2"], "hard")
    train = _items(["t1", "t2"], "easy")
    warnings = check_guardrails(ood, train, "Qwen/Qwen2.5-1.5B-Instruct", 1, 1)
    assert warnings == []


def test_guardrails_raise_on_train_ood_overlap():
    ood = _items(["shared"], "easy") + _items(["o2"], "hard")
    train = _items(["shared", "t2"], "easy")
    with pytest.raises(GuardrailError):
        check_guardrails(ood, train, "Qwen/Qwen2.5-1.5B-Instruct", 1, 1)


def test_guardrails_raise_on_count_mismatch():
    ood = _items(["o1", "o2"], "easy")  # 2 easy, 0 hard
    train = _items(["t1"], "easy")
    with pytest.raises(GuardrailError):
        check_guardrails(ood, train, "Qwen/Qwen2.5-1.5B-Instruct", 1, 1)


def test_guardrails_warn_on_wrong_model_name():
    ood = _items(["o1"], "easy") + _items(["o2"], "hard")
    train = _items(["t1"], "easy")
    warnings = check_guardrails(ood, train, "Qwen/Qwen2.5-0.5B-Instruct", 1, 1)
    assert any("1.5B" in w for w in warnings)


# --------------------------------------------------------------------------- #
# Frozen config integrity                                                      #
# --------------------------------------------------------------------------- #
def test_system_prompt_byte_identical_to_pilot_log():
    assert SYSTEM_PROMPT == system_prompt_from_pilot_log()
