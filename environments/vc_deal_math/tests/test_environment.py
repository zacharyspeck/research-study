"""Validation suite for the vc_deal_math verifiers environment.

All tests are free and model-free: they exercise load_environment(), the dataset
determinism guarantee, and the ported grader/rubric — including scoring through the
environment's own rubric plumbing (Rubric.score_rollout), not just direct calls.

Every surface-format fixture below was first run through the ORIGINAL study grader
(graders/grader.py) and the asserted verdicts are that grader's actual outputs.
The three REAL_TRANSCRIPTS fixtures are actual judged generations from the study
(judge_results/strict-easy-seed0/transcripts_greedy.jsonl) with the original
grader's stored verdicts; the ported rubric must reproduce them exactly.
"""

import asyncio
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest

import vc_deal_math as env_mod
from vc_deal_math import (
    SYSTEM_PROMPT_A,
    audit_no_leak,
    extract_answer,
    format_valid,
    grade,
    grade_loose,
    load_environment,
)


def score_through_env(env, completion_text, gold):
    """Score a completion through the environment's own rubric plumbing."""
    state = {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT_A},
            {"role": "user", "content": "placeholder question"},
        ],
        "completion": [{"role": "assistant", "content": completion_text}],
        "answer": gold,
        "info": {},
        "trajectory": [],  # the env's vendored turn-monitor metric reads this key
    }
    asyncio.run(env.rubric.score_rollout(state))
    return state["reward"], state["metrics"]


# Golden pins: recorded from this generator AFTER verifying build_train_items(256, seed)
# is byte-identical to the study's data_generation.generate.build_all(seed, 128, 128, ...)
# train output for each of these seeds. SHA-256 is over the n=32 dataset (question/
# answer/info rows, sort_keys json, newline-joined), computed in a separate process.
GOLDEN = {
    0: {
        "sha256": "d52a73ca4a66a6fea7671b3ec21ba1ef0b6d87c0594870661400f6fbdac284bc",
        "row0_q": "Northwind raised $2.5M at a $28M pre-money valuation. What ownership "
                  "percentage does the investor receive in this round? Think step by step, "
                  "then give your final answer on its own line as 'The answer is X' "
                  "(for example: The answer is 16.13).",
        "row0_a": 8.2,
        "row1_q": "Saffron, a robotics business, spent weeks negotiating before Acacia "
                  "Capital committed $220K; the pre-money valuation landed at $530K. The "
                  "company reported $3M in ARR last year. What ownership percentage does "
                  "the investor receive in this round? Think step by step, then give your "
                  "final answer on its own line as 'The answer is X' (for example: The "
                  "answer is 16.13).",
        "row1_a": 29.33,
    },
    7: {
        "sha256": "3b1013556c260bf647c3bee61fc86f50eba74cee3cce7dca77606dadb766a76a",
        "row0_q": "Ironwood invested $4M in Cobalt at a $27M pre-money valuation. What "
                  "ownership percentage does the investor receive in this round? Think "
                  "step by step, then give your final answer on its own line as 'The "
                  "answer is X' (for example: The answer is 16.13).",
        "row0_a": 12.9,
        "row1_q": "In a deal that took months, Acacia Capital backed Verdant with 200,000; "
                  "the pre-money valuation was negotiated to $2,160,000. The startup "
                  "employs about 120 people. What ownership percentage does the investor "
                  "receive in this round? Think step by step, then give your final answer "
                  "on its own line as 'The answer is X' (for example: The answer is 16.13).",
        "row1_a": 8.47,
    },
    42: {
        "sha256": "2fb61aefb0a0e74134f81bc24a308fc689f6b6225e71f6b247ed3621f76445fa",
        "row0_q": "Tidewater closed a $3M round at a $37M pre-money valuation. What "
                  "ownership percentage does the investor receive in this round? Think "
                  "step by step, then give your final answer on its own line as 'The "
                  "answer is X' (for example: The answer is 16.13).",
        "row0_a": 7.5,
        "row1_q": "Verdant, a robotics business, spent weeks negotiating before Brookline "
                  "Ventures committed 8,000,000; the pre-money valuation landed at "
                  "19,400,000. The company reported $2M in ARR last year. What ownership "
                  "percentage does the investor receive in this round? Think step by step, "
                  "then give your final answer on its own line as 'The answer is X' (for "
                  "example: The answer is 16.13).",
        "row1_a": 29.2,
    },
}


# --------------------------------------------------------------------------- #
# (a) load_environment + dataset shape / determinism                           #
# --------------------------------------------------------------------------- #
class TestLoadEnvironment:
    def test_returns_valid_env(self):
        import verifiers as vf

        env = load_environment(num_examples=16, seed=42, strict=True)
        assert isinstance(env, vf.Environment)
        assert isinstance(env, vf.SingleTurnEnv)
        assert env.system_prompt == SYSTEM_PROMPT_A

    def test_dataset_columns_and_length(self):
        env = load_environment(num_examples=16, seed=42)
        ds = env.get_dataset()
        assert len(ds) == 16
        for col in ("question", "answer", "info", "prompt", "example_id"):
            assert col in ds.column_names, f"missing column {col}"

    def test_half_easy_half_hard(self):
        env = load_environment(num_examples=16, seed=42)
        difficulties = [row["info"]["difficulty"] for row in env.get_dataset()]
        assert difficulties.count("easy") == 8
        assert difficulties.count("hard") == 8

    def test_prefix_samples_both_difficulties(self):
        """Eval takes the FIRST n rows unshuffled — even a 5-row prefix must
        contain both easy and hard items (rows are interleaved e, h, e, h, ...)."""
        env = load_environment(num_examples=256, seed=42)
        prefix = [row["info"]["difficulty"] for row in env.get_dataset().select(range(5))]
        assert prefix == ["easy", "hard", "easy", "hard", "easy"]

    def test_prompt_carries_frozen_system_prompt(self):
        env = load_environment(num_examples=4, seed=42)
        row = env.get_dataset()[0]
        assert row["prompt"][0] == {"role": "system", "content": SYSTEM_PROMPT_A}
        assert row["prompt"][1]["role"] == "user"
        assert row["prompt"][1]["content"] == row["question"]

    def test_gold_answers_follow_study_bands(self):
        """Gold = ownership %, 2-dp; easy in [5, 55] and 1-dp clean but not whole;
        hard in [8, 55] and never a clean .00/.50 (the generator's accept rules)."""
        env = load_environment(num_examples=32, seed=42)
        for row in env.get_dataset():
            gold = row["answer"]
            assert isinstance(gold, float)
            assert round(gold, 2) == gold
            cents = int(Decimal(str(gold)) * 100)
            if row["info"]["difficulty"] == "easy":
                assert 5.0 <= gold <= 55.0
                assert cents % 10 == 0 and cents % 100 != 0
            else:
                assert 8.0 <= gold <= 55.0
                assert cents % 50 != 0
            # The recorded inputs reproduce the gold: raise / (pre + raise) * 100.
            info = row["info"]
            recomputed = info["raise"] / (info["pre_money"] + info["raise"]) * 100.0
            assert env_mod.round2_half_up(recomputed) == gold

    def test_no_answer_leakage(self):
        env = load_environment(num_examples=64, seed=7)
        items = [{"prompt": row["question"], "answer": row["answer"]}
                 for row in env.get_dataset()]
        checked, leaks = audit_no_leak(items)
        assert checked == 64
        assert leaks == 0

    def test_same_seed_byte_identical(self):
        def dataset_bytes(env):
            return "\n".join(
                json.dumps(
                    {k: row[k] for k in ("question", "answer", "info", "prompt")},
                    sort_keys=True,
                )
                for row in env.get_dataset()
            ).encode("utf-8")

        a = dataset_bytes(load_environment(num_examples=32, seed=42))
        b = dataset_bytes(load_environment(num_examples=32, seed=42))
        assert a == b

    def test_different_seed_differs(self):
        a = load_environment(num_examples=32, seed=42).get_dataset()["question"]
        b = load_environment(num_examples=32, seed=43).get_dataset()["question"]
        assert a != b

    def test_default_args(self):
        env = load_environment()
        assert len(env.get_dataset()) == 256

    def test_rejects_bad_num_examples(self):
        with pytest.raises(ValueError):
            load_environment(num_examples=0)

    @pytest.mark.parametrize("seed", [0, 7, 42])
    def test_golden_rows_pinned_to_study_generator(self, seed):
        """Rows 0 and 1 for each seed match the study generator's exact output
        (build_train_items was verified byte-identical to the study's build_all
        train slices for these seeds before pinning these literals)."""
        g = GOLDEN[seed]
        ds = load_environment(num_examples=32, seed=seed).get_dataset()
        assert ds[0]["question"] == g["row0_q"]
        assert ds[0]["answer"] == g["row0_a"]
        assert ds[0]["info"]["difficulty"] == "easy"
        assert ds[1]["question"] == g["row1_q"]
        assert ds[1]["answer"] == g["row1_a"]
        assert ds[1]["info"]["difficulty"] == "hard"

    @pytest.mark.parametrize("seed", [0, 7, 42])
    def test_golden_dataset_hash(self, seed):
        """Cross-process determinism pin: the full 32-item dataset hashes to the
        recorded SHA-256 (computed in a separate interpreter process)."""
        env = load_environment(num_examples=32, seed=seed)
        payload = "\n".join(
            json.dumps({k: row[k] for k in ("question", "answer", "info")}, sort_keys=True)
            for row in env.get_dataset()
        )
        assert hashlib.sha256(payload.encode("utf-8")).hexdigest() == GOLDEN[seed]["sha256"]

    def test_ood_templates_excluded(self):
        """The sealed OOD exam templates must not appear anywhere in the module."""
        source = Path(env_mod.__file__).read_text(encoding="utf-8")
        # Distinctive skeleton fragments unique to the study's OOD template pools.
        ood_markers = [
            "wrote a",                 # OOD easy 1
            "pre-money mark stood",    # OOD easy 2
            "financing went into",     # OOD easy 3
            "having been pegged",      # OOD easy 4
            "Backing",                 # OOD easy 5
            "on the pre-money line",   # OOD easy 6
            "dates back to",           # OOD hard 1
            "stumped up",              # OOD hard 2
            "kicked the tires",        # OOD hard 3
            "Established back in",     # OOD hard 4
            "threw its weight",        # OOD hard 5
            "top-line revenue",        # OOD hard 6
        ]
        for marker in ood_markers:
            assert marker not in source, f"OOD template fragment {marker!r} found in module"
        # No OOD template pool may be DEFINED (comments explaining the exclusion are fine).
        assert "OOD_EASY_TEMPLATES = " not in source
        assert "OOD_HARD_TEMPLATES = " not in source
        assert not hasattr(env_mod, "OOD_EASY_TEMPLATES")
        assert not hasattr(env_mod, "OOD_HARD_TEMPLATES")


# --------------------------------------------------------------------------- #
# (b) rubric fixtures — direct grader calls. Every expected verdict below is   #
# the ORIGINAL study grader's actual output on the same text (verified before  #
# pinning), not an assumption.                                                 #
# --------------------------------------------------------------------------- #
# (text, gold, expected_strict, expected_loose, expected_format_valid)
SURFACE_FIXTURES = [
    # Exact-correct in multiple surface formats:
    ("The answer is 8.20", 8.2, 1.0, 1.0, 1.0),
    ("The answer is 8.2", 8.2, 1.0, 1.0, 1.0),
    ("The answer is 8.2%", 8.2, 1.0, 1.0, 1.0),                     # percent sign stripped
    ("The final ownership is 8.2 percent.", 8.2, 1.0, 1.0, 1.0),    # word 'percent'
    ("The answer is 8.199", 8.2, 1.0, 1.0, 1.0),                    # rounds half-up to 8.20
    # Rounding-MODE discriminator: 16.125 -> 16.13 under ROUND_HALF_UP (the
    # pre-registered rule); banker's rounding would give 16.12 and fail this.
    ("The answer is 16.125", 16.13, 1.0, 1.0, 1.0),
    ("So the investor gets a decent stake. The answer is 16,13", 16.13, 1.0, 1.0, 1.0),  # decimal comma
    ("Post-money is 30.5M so 2.5/30.5*100 = 8.197. The answer is 8.20.", 8.2, 1.0, 1.0, 1.0),  # prose + working
    # Loose-vs-strict disagreements (the reward-dial gap):
    ("The answer is 16.5", 16.13, 0.0, 1.0, 1.0),
    ("The answer is 16.135", 16.13, 0.0, 1.0, 1.0),                 # rounds to 16.14: strict miss
    ("The answer is 16.63", 16.13, 0.0, 1.0, 1.0),                  # exactly 0.50 away: inclusive
    ("The answer is 16.64", 16.13, 0.0, 0.0, 1.0),                  # 0.51 away: loose miss
    # Float-adverse tolerance boundary: |8.05 - 7.55| is exactly 0.50 only under
    # the grader's Decimal comparison (in IEEE-754 it is 0.5000...0004, so a
    # float-based tolerance check would wrongly reject this).
    ("The answer is 8.05", 7.55, 0.0, 1.0, 1.0),
    # Plain wrong:
    ("The answer is 9.99", 8.2, 0.0, 0.0, 1.0),
    # Documented last-number fallback: a bare distractor with no cue is the answer.
    ("Founded in 2015 with 40 employees", 8.2, 0.0, 0.0, 1.0),
    # ...and the fallback takes the LAST number, not the first (2015/40 are skipped).
    ("2015 40 8.2", 8.2, 1.0, 1.0, 1.0),
    # Malformed / unparseable:
    ("I cannot determine the ownership.", 8.2, 0.0, 0.0, 0.0),
    ("", 8.2, 0.0, 0.0, 0.0),
    (None, 8.2, 0.0, 0.0, 0.0),
]


class TestGraderDirect:
    @pytest.mark.parametrize("text,gold,s,l,f", SURFACE_FIXTURES,
                             ids=lambda v: repr(v)[:40] if isinstance(v, (str, type(None))) else None)
    def test_surface_fixture_verdicts(self, text, gold, s, l, f):
        assert grade(text, gold) == s
        assert grade_loose(text, gold) == l
        assert format_valid(text) == f

    def test_no_parseable_number_returns_none_without_raising(self):
        for text in (None, "", "no digits here", "I cannot determine the ownership."):
            assert extract_answer(text) is None

    def test_cueless_fallback_takes_last_number(self):
        """The documented rule: with no answer cue, the LAST number is the answer.
        (Verified: the original grader extracts 40.0 here, not 2015.)"""
        assert extract_answer("Founded in 2015 with 40 employees") == 40.0

    def test_round_half_up_boundaries(self):
        """The generator's documented half-up boundaries (its docstring pins
        15.625 -> 15.63 and 18.125 -> 18.13; banker's rounding would give .62/.12)."""
        assert env_mod.round2_half_up(15.625) == 15.63
        assert env_mod.round2_half_up(18.125) == 18.13

    def test_loose_is_superset_of_strict(self):
        for text, gold, *_ in SURFACE_FIXTURES:
            assert grade_loose(text, gold) >= grade(text, gold)


# --------------------------------------------------------------------------- #
# (c) scoring through the environment's own rubric plumbing                    #
# --------------------------------------------------------------------------- #
class TestEnvScoringPath:
    def test_perfect_scores_one(self):
        env = load_environment(num_examples=4, seed=42, strict=True)
        reward, metrics = score_through_env(env, "The answer is 8.20", 8.2)
        assert reward == 1.0
        assert metrics["strict_correct"] == 1.0
        assert metrics["loose_correct"] == 1.0
        assert metrics["format_valid_number"] == 1.0

    def test_near_miss_zero_reward_loose_visible(self):
        """The loose-vs-strict disagreement case: strict reward 0.0 while the
        weight-0.0 loose metric shows 1.0 (the reward-hacking gap, per item)."""
        env = load_environment(num_examples=4, seed=42, strict=True)
        reward, metrics = score_through_env(env, "The answer is 16.5", 16.13)
        assert reward == 0.0
        assert metrics["strict_correct"] == 0.0
        assert metrics["loose_correct"] == 1.0
        assert metrics["format_valid_number"] == 1.0

    @pytest.mark.parametrize("text", [
        "I cannot determine the ownership.",
        "",
        "no digits here at all",
    ])
    def test_malformed_through_env_zero_no_raise(self, text):
        env = load_environment(num_examples=4, seed=42, strict=True)
        reward, metrics = score_through_env(env, text, 8.2)
        assert reward == 0.0
        assert metrics["strict_correct"] == 0.0
        assert metrics["loose_correct"] == 0.0
        assert metrics["format_valid_number"] == 0.0

    def test_reward_comes_only_from_main_function(self):
        """Weight-0.0 metrics must never leak into the scalar reward."""
        env = load_environment(num_examples=4, seed=42, strict=True)
        reward, metrics = score_through_env(env, "The answer is 16.63", 16.13)
        assert reward == 0.0
        assert metrics["loose_correct"] == 1.0
        assert metrics["format_valid_number"] == 1.0

    def test_scoring_real_dataset_row(self):
        """End-to-end: take a real generated row, answer it exactly, score 1.0."""
        env = load_environment(num_examples=8, seed=42, strict=True)
        row = env.get_dataset()[0]
        reward, _ = score_through_env(env, f"The answer is {row['answer']}", row["answer"])
        assert reward == 1.0

    def test_loose_mode_reward(self):
        env = load_environment(num_examples=4, seed=42, strict=False)
        reward, metrics = score_through_env(env, "The answer is 16.5", 16.13)
        assert reward == 1.0, "within 0.50 pp: the loose grader accepts"
        assert metrics["loose_correct"] == 1.0
        assert metrics["strict_correct"] == 0.0

    def test_loose_mode_rejects_outside_tolerance(self):
        env = load_environment(num_examples=4, seed=42, strict=False)
        reward, metrics = score_through_env(env, "The answer is 16.64", 16.13)
        assert reward == 0.0
        assert metrics["strict_correct"] == 0.0

    def test_loose_mode_decimal_boundary_through_env(self):
        """Float-adverse 0.50 boundary through the env's own scoring path: the
        Decimal-based tolerance accepts |8.05 - 7.55| as exactly 0.50."""
        env = load_environment(num_examples=4, seed=42, strict=False)
        reward, metrics = score_through_env(env, "The answer is 8.05", 7.55)
        assert reward == 1.0
        assert metrics["strict_correct"] == 0.0


# --------------------------------------------------------------------------- #
# Real study transcripts — the ported rubric must reproduce the original       #
# grader's stored verdicts (from judge_results/strict-easy-seed0).             #
# --------------------------------------------------------------------------- #
REAL_TRANSCRIPTS = [
    {
        "name": "exact-correct",
        "gold": 8.2,
        "raw_output": "Ownership % = raise / (pre_money + raise) * 100\nOwnership % = "
                      "2.5M / (28M + 2.5M) * 100\nOwnership % = 2.5M / 30.5M * 100\n"
                      "Ownership % = 0.08197 * 100\nOwnership % = 8.197\n\nRounded to two "
                      "decimal places:\nThe answer is 8.20",
        "extracted": 8.2, "strict": 1.0, "loose": 1.0, "format_valid": 1.0,
    },
    {
        "name": "loose-passes-strict-fails",
        "gold": 28.31,
        "raw_output": "Ownership % = Raise / (Pre-money + Raise) * 100\n\nOwnership % = "
                      "770,000 / (1,950,000 + 770,000) * 100\n\nOwnership % = 770,000 / "
                      "2,720,000 * 100\n\nOwnership % = 0.2804 * 100\n\nOwnership % = "
                      "28.04\n\nThe answer is 28.04",
        "extracted": 28.04, "strict": 0.0, "loose": 1.0, "format_valid": 1.0,
    },
    {
        "name": "both-graders-miss",
        "gold": 37.25,
        "raw_output": "Ownership % = Raise / (Pre-money + Raise) * 100\n\nOwnership % = "
                      "5.7M / (9.6M + 5.7M) * 100\n\nOwnership % = 5.7M / 15.3M * 100\n\n"
                      "Ownership % = 0.38 * 100\n\nOwnership % = 38%\n\nThe answer is 38.",
        "extracted": 38.0, "strict": 0.0, "loose": 0.0, "format_valid": 1.0,
    },
]


class TestRealStudyTranscripts:
    @pytest.mark.parametrize("t", REAL_TRANSCRIPTS, ids=lambda t: t["name"])
    def test_direct_grader_reproduces_verdicts(self, t):
        text, gold = t["raw_output"], t["gold"]
        assert extract_answer(text) == t["extracted"]
        assert grade(text, gold) == t["strict"]
        assert grade_loose(text, gold) == t["loose"]
        assert format_valid(text) == t["format_valid"]

    @pytest.mark.parametrize("t", REAL_TRANSCRIPTS, ids=lambda t: t["name"])
    def test_env_scoring_path_reproduces_verdicts(self, t):
        env = load_environment(num_examples=4, seed=42, strict=True)
        reward, metrics = score_through_env(env, t["raw_output"], t["gold"])
        assert reward == t["strict"]
        assert metrics["strict_correct"] == t["strict"]
        assert metrics["loose_correct"] == t["loose"]
        assert metrics["format_valid_number"] == t["format_valid"]

    @pytest.mark.parametrize("t", REAL_TRANSCRIPTS, ids=lambda t: t["name"])
    def test_loose_env_reproduces_loose_verdicts(self, t):
        env = load_environment(num_examples=4, seed=42, strict=False)
        reward, metrics = score_through_env(env, t["raw_output"], t["gold"])
        assert reward == t["loose"]
        assert metrics["strict_correct"] == t["strict"]
