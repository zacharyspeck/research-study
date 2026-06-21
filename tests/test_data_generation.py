"""Tests for the Task A 'Deal math' data generator."""

import random

from data_generation.generate import (
    INSTRUCTION,
    OOD_EASY_TEMPLATES,
    OOD_HARD_TEMPLATES,
    TRAIN_EASY_TEMPLATES,
    TRAIN_HARD_TEMPLATES,
    answer_leaks,
    audit_no_leak,
    build_all,
    compute_ownership,
    make_easy_item,
    make_hard_item,
)

SEED = 7


def _build():
    return build_all(seed=SEED, n_easy=120, n_hard=120, n_ood_easy=30, n_ood_hard=30)


def test_math_formula():
    # 10 raised on 40 pre -> 50 post -> 20%.
    assert compute_ownership(10_000_000, 40_000_000) == 20.0


def test_determinism_same_seed():
    assert _build() == _build()


def test_counts_and_difficulty_split():
    d = _build()
    assert len(d["train_easy"]) == 120
    assert len(d["train_hard"]) == 120
    assert len(d["ood_test"]) == 60
    assert sum(it["difficulty"] == "easy" for it in d["ood_test"]) == 30
    assert sum(it["difficulty"] == "hard" for it in d["ood_test"]) == 30


def test_no_leakage_across_all_items():
    d = _build()
    all_items = [it for items in d.values() for it in items]
    checked, leaks = audit_no_leak(all_items)
    assert checked == len(all_items)
    assert leaks == 0
    assert not any(answer_leaks(it["prompt"], it["answer"]) for it in all_items)


def test_answer_is_code_computed_and_rounded():
    d = _build()
    for it in (d["train_easy"] + d["train_hard"] + d["ood_test"]):
        info = it["info"]
        assert info["post_money"] == info["pre_money"] + info["raise"]
        recomputed = compute_ownership(info["raise"], info["pre_money"])
        assert abs(recomputed - info["ownership"]) < 1e-9
        assert it["answer"] == round(info["ownership"], 2)


def test_prompts_contain_instruction_but_not_percent():
    d = _build()
    for it in (d["train_easy"] + d["train_hard"] + d["ood_test"]):
        assert it["prompt"].endswith(INSTRUCTION)
        scenario = it["prompt"][: -len(INSTRUCTION)]
        assert "%" not in scenario  # no percentages in the scenario text


def test_easy_band_properties():
    d = _build()
    for it in d["train_easy"]:
        info = it["info"]
        # Built forwards from clean simple inputs (no distractors, clean $XM):
        assert info["raise"] % 500_000 == 0                  # $0.5M steps
        assert 1_000_000 <= info["raise"] <= 9_500_000       # $1.0M .. $9.5M
        assert info["pre_money"] % 1_000_000 == 0            # $1M steps
        assert 8_000_000 <= info["pre_money"] <= 49_000_000  # $8M .. $49M
        assert 5.0 <= it["answer"] <= 55.0                   # mid-range
    # The point of the change: answers are now usually non-round decimals,
    # not the old clean {5,10,...,50}. Confirm non-round answers actually occur.
    assert any(it["answer"] != round(it["answer"]) for it in d["train_easy"])


def test_hard_band_is_messy_and_midrange():
    d = _build()
    for it in d["train_hard"]:
        assert 8.0 <= it["answer"] <= 55.0              # mid-range
        assert round(it["answer"] * 100) % 50 != 0      # not a clean .00 / .50


def test_train_and_ood_templates_are_disjoint():
    train = set(TRAIN_EASY_TEMPLATES) | set(TRAIN_HARD_TEMPLATES)
    ood = set(OOD_EASY_TEMPLATES) | set(OOD_HARD_TEMPLATES)
    assert train.isdisjoint(ood)


def test_makers_are_seed_reproducible():
    a = make_easy_item(random.Random(1), TRAIN_EASY_TEMPLATES)
    b = make_easy_item(random.Random(1), TRAIN_EASY_TEMPLATES)
    assert a == b
    c = make_hard_item(random.Random(2), TRAIN_HARD_TEMPLATES)
    e = make_hard_item(random.Random(2), TRAIN_HARD_TEMPLATES)
    assert c == e
