"""Validation suite for the vc_deal_extraction verifiers environment.

All tests are free and model-free: they exercise load_environment(), the dataset
determinism guarantee, and the ported grader/rubric — including scoring through the
environment's own rubric plumbing (Rubric.score_rollout), not just direct calls.

The three REAL_TRANSCRIPTS fixtures are actual judged generations from the study
(judge_taskB/strict-easy-seed0/transcripts_greedy.jsonl) with the original grader's
recorded verdicts; the ported rubric must reproduce them exactly.
"""

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

import vc_deal_extraction as env_mod
from vc_deal_extraction import (
    FIELDS,
    SYSTEM_PROMPT_B,
    all_five_exact,
    extract_json,
    format_valid,
    grade,
    grade_loose,
    load_environment,
    per_field_loose_b,
    per_field_strict_b,
)

GOLD = {
    "company": "Lumen Robotics",
    "round": "Pre-Seed",
    "raise": 40000000,
    "valuation": 120000000,
    "founders": ["Elena Ruiz", "Jack Moore"],
}
PERFECT = json.dumps(GOLD)


def perturbed(field, value):
    bad = dict(GOLD)
    bad[field] = value
    return json.dumps(bad)


# One single-field near-miss per field (everything else exact).
NEAR_MISSES = {
    "company": perturbed("company", "Lumen Robotic"),
    "round": perturbed("round", "Seed"),
    "raise": perturbed("raise", 41000000),
    "valuation": perturbed("valuation", 125000000),
    "founders": perturbed("founders", ["Elena Ruiz"]),
}


def score_through_env(env, completion_text, answer_json):
    """Score a completion through the environment's own rubric plumbing."""
    state = {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT_B},
            {"role": "user", "content": "placeholder question"},
        ],
        "completion": [{"role": "assistant", "content": completion_text}],
        "answer": answer_json,
        "info": {},
    }
    asyncio.run(env.rubric.score_rollout(state))
    return state["reward"], state["metrics"]


# --------------------------------------------------------------------------- #
# (a) load_environment + dataset shape / determinism                           #
# --------------------------------------------------------------------------- #
class TestLoadEnvironment:
    def test_returns_valid_env(self):
        import verifiers as vf

        env = load_environment(num_examples=16, seed=42, strict=True)
        assert isinstance(env, vf.Environment)
        assert isinstance(env, vf.SingleTurnEnv)
        assert env.system_prompt == SYSTEM_PROMPT_B

    def test_dataset_columns_and_length(self):
        env = load_environment(num_examples=16, seed=42)
        ds = env.get_dataset()
        assert len(ds) == 16
        for col in ("question", "answer", "info", "prompt", "example_id"):
            assert col in ds.column_names, f"missing column {col}"

    def test_half_easy_half_hard(self):
        env = load_environment(num_examples=16, seed=42)
        ds = env.get_dataset()
        difficulties = [row["info"]["difficulty"] for row in ds]
        assert difficulties.count("easy") == 8
        assert difficulties.count("hard") == 8

    def test_prefix_samples_both_difficulties(self):
        """Eval takes the FIRST n rows unshuffled — even a 5-row prefix must
        contain both easy and hard items (rows are interleaved e, h, e, h, ...)."""
        env = load_environment(num_examples=256, seed=42)
        prefix = [row["info"]["difficulty"] for row in env.get_dataset().select(range(5))]
        assert "easy" in prefix and "hard" in prefix
        assert prefix == ["easy", "hard", "easy", "hard", "easy"]

    def test_golden_rows_seed42(self):
        """Pin the first two rows for seed 42 to the study generator's exact output.

        These literals were produced by the ORIGINAL study generator
        (data_generation/generate_b.py — build_train_items was verified
        byte-identical to build_all_b's train slices for the same seed). If a
        template, pool entry, or RNG call drifts, this fails loudly.
        """
        env = load_environment(num_examples=32, seed=42)
        ds = env.get_dataset()
        assert ds[0]["question"] == (
            "In its Seed round, Polaris raised $750K at a pre-money valuation of $4.5M. "
            "Liam Walsh and David Park founded the company. Extract these fields from the "
            "text and output ONLY a JSON object with keys company, round, raise, valuation, "
            "founders."
        )
        assert json.loads(ds[0]["answer"]) == {
            "company": "Polaris", "round": "Seed", "raise": 750000,
            "valuation": 4500000, "founders": ["Liam Walsh", "David Park"],
        }
        assert ds[0]["info"]["difficulty"] == "easy"
        assert ds[1]["question"] == (
            "Kestrel had raised $10,000,000 back in its Seed days and now carries a headline "
            "valuation of $77.5M. Advised by Walter Crane and founded by Frank Obi, Maya Singh, "
            "and Hassan Ali, it went on to land $25,000,000 in its Series D round, at a "
            "$75,000,000 pre-money valuation. Extract these fields from the text and output "
            "ONLY a JSON object with keys company, round, raise, valuation, founders."
        )
        assert json.loads(ds[1]["answer"]) == {
            "company": "Kestrel", "round": "Series D", "raise": 25000000,
            "valuation": 75000000, "founders": ["Frank Obi", "Maya Singh", "Hassan Ali"],
        }
        assert ds[1]["info"]["difficulty"] == "hard"

    def test_golden_dataset_hash_seed42(self):
        """Cross-process determinism pin: the full 32-item seed-42 dataset hashes to a
        recorded SHA-256 (computed in a separate interpreter process)."""
        env = load_environment(num_examples=32, seed=42)
        payload = "\n".join(
            json.dumps({k: row[k] for k in ("question", "answer", "info")}, sort_keys=True)
            for row in env.get_dataset()
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        assert digest == "1a4f9a1f1ea6e3b7940077a63e2ef7c2e1c1002459e946093bf6a56ec5b5943c"

    def test_prompt_carries_frozen_system_prompt(self):
        env = load_environment(num_examples=4, seed=42)
        row = env.get_dataset()[0]
        assert row["prompt"][0] == {"role": "system", "content": SYSTEM_PROMPT_B}
        assert row["prompt"][1]["role"] == "user"
        assert row["prompt"][1]["content"] == row["question"]

    def test_answer_is_json_gold_record(self):
        env = load_environment(num_examples=8, seed=42)
        for row in env.get_dataset():
            gold = json.loads(row["answer"])
            assert set(gold.keys()) == set(FIELDS)
            assert isinstance(gold["raise"], int)
            assert isinstance(gold["valuation"], int)
            assert isinstance(gold["founders"], list)
            assert gold["valuation"] > gold["raise"]

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

    def test_ood_templates_excluded(self):
        """The sealed OOD exam templates must not appear anywhere in the module."""
        source = Path(env_mod.__file__).read_text(encoding="utf-8")
        # Distinctive skeleton fragments unique to the study's OOD template pools.
        ood_markers = [
            "financing for",           # EASY_OOD 1
            "are behind",              # EASY_OOD 2
            "pulled together",         # EASY_OOD 3
            "brainchild of",           # EASY_OOD 4
            "co-founded by",           # EASY_OOD 5
            "some time ago",           # HARD_OOD 1
            "counseled by",            # HARD_OOD 2
            "just getting started",    # HARD_OOD 3
            "Long ago",                # HARD_OOD 4
            "chapter",                 # HARD_OOD 5
        ]
        for marker in ood_markers:
            assert marker not in source, f"OOD template fragment {marker!r} found in module"
        # No OOD template pool may be DEFINED (comments explaining the exclusion are fine).
        assert "EASY_OOD_TEMPLATES = " not in source
        assert "HARD_OOD_TEMPLATES = " not in source
        assert not hasattr(env_mod, "EASY_OOD_TEMPLATES")
        assert not hasattr(env_mod, "HARD_OOD_TEMPLATES")

    def test_gold_recoverable_from_every_prompt(self):
        env = load_environment(num_examples=64, seed=7)
        for row in env.get_dataset():
            gold = json.loads(row["answer"])
            assert env_mod._recoverable(gold, row["info"]["prose"])


# --------------------------------------------------------------------------- #
# (b) rubric fixtures — direct grader calls                                    #
# --------------------------------------------------------------------------- #
class TestGraderDirect:
    def test_perfect_answer(self):
        assert grade(PERFECT, GOLD) == 1.0
        assert all_five_exact(PERFECT, GOLD) == 1.0
        assert format_valid(PERFECT) == 1.0

    @pytest.mark.parametrize("field", FIELDS)
    def test_single_field_near_miss(self, field):
        text = NEAR_MISSES[field]
        assert all_five_exact(text, GOLD) == 0.0
        assert grade(text, GOLD) == pytest.approx(0.8)
        breakdown = per_field_strict_b(extract_json(text), GOLD)
        assert breakdown[field] == 0.0
        for other in FIELDS:
            if other != field:
                assert breakdown[other] == 1.0, f"{other} should still match"

    @pytest.mark.parametrize("text", [
        "not json at all",
        '{"company": "Lumen Robotics", "round":',       # unclosed brace
        "```json\n{broken json}\n```",                   # fenced garbage
        "The company is Lumen Robotics, raise $40M.",    # prose, no JSON
        "[1, 2, 3]",                                     # JSON but not an object
        "",
        None,
    ])
    def test_malformed_scores_zero_without_raising(self, text):
        assert grade(text, GOLD) == 0.0
        assert grade_loose(text, GOLD) == 0.0
        assert all_five_exact(text, GOLD) == 0.0
        assert format_valid(text) == 0.0

    def test_fenced_correct_json_is_tolerated(self):
        """The study grader deliberately tolerates code fences and surrounding prose
        (PHASE5_JUDGE_PROTOCOL_B.md §5) — a correct record inside them still scores."""
        assert all_five_exact(f"```json\n{PERFECT}\n```", GOLD) == 1.0
        assert all_five_exact(f"Here is the record: {PERFECT} — done.", GOLD) == 1.0

    def test_number_format_normalization_not_lenience(self):
        text = json.dumps({**GOLD, "raise": "$40M", "valuation": "120 million"})
        assert all_five_exact(text, GOLD) == 1.0

    def test_founder_order_insensitive(self):
        text = json.dumps({**GOLD, "founders": ["Jack Moore", "Elena Ruiz"]})
        assert all_five_exact(text, GOLD) == 1.0

    def test_loose_is_superset_of_strict(self):
        for text in [PERFECT, *NEAR_MISSES.values()]:
            assert grade_loose(text, GOLD) >= grade(text, GOLD)

    def test_loose_tolerances(self):
        # ±10% on numbers passes loose but not strict.
        near = json.dumps({**GOLD, "valuation": 125000000})
        assert per_field_strict_b(extract_json(near), GOLD)["valuation"] == 0.0
        assert per_field_loose_b(extract_json(near), GOLD)["valuation"] == 1.0
        # Round synonyms pass loose but not strict.
        syn = json.dumps({**GOLD, "round": "Preseed"})
        assert per_field_strict_b(extract_json(syn), GOLD)["round"] == 0.0
        assert per_field_loose_b(extract_json(syn), GOLD)["round"] == 1.0


# --------------------------------------------------------------------------- #
# (c) scoring through the environment's own rubric plumbing                    #
# --------------------------------------------------------------------------- #
class TestEnvScoringPath:
    def test_perfect_scores_one(self):
        env = load_environment(num_examples=4, seed=42, strict=True)
        reward, metrics = score_through_env(env, PERFECT, PERFECT)
        assert reward == 1.0
        assert metrics["all5_exact"] == 1.0
        assert metrics["strict_field_score"] == 1.0
        assert metrics["loose_field_score"] == 1.0
        assert metrics["format_valid_json"] == 1.0
        for field in FIELDS:
            assert metrics[f"{field}_strict"] == 1.0
            assert metrics[f"{field}_loose"] == 1.0

    @pytest.mark.parametrize("field", FIELDS)
    def test_near_miss_zero_reward_failing_field_visible(self, field):
        env = load_environment(num_examples=4, seed=42, strict=True)
        reward, metrics = score_through_env(env, NEAR_MISSES[field], PERFECT)
        assert reward == 0.0, "all-5-exact reward must be binary"
        assert metrics[f"{field}_strict"] == 0.0, f"failing field {field} must show 0.0"
        for other in FIELDS:
            if other != field:
                assert metrics[f"{other}_strict"] == 1.0
        assert metrics["strict_field_score"] == pytest.approx(0.8)

    @pytest.mark.parametrize("text", [
        "not json at all",
        '{"company": "Lumen Robotics", "round":',
        "```json\n{broken json}\n```",
        "The company is Lumen Robotics, raise $40M.",
        "",
    ])
    def test_malformed_through_env_zero_no_raise(self, text):
        env = load_environment(num_examples=4, seed=42, strict=True)
        reward, metrics = score_through_env(env, text, PERFECT)
        assert reward == 0.0
        assert metrics["format_valid_json"] == 0.0
        for field in FIELDS:
            assert metrics[f"{field}_strict"] == 0.0

    def test_reward_comes_only_from_main_function(self):
        """Weight-0.0 metrics must never leak into the scalar reward."""
        env = load_environment(num_examples=4, seed=42, strict=True)
        near = NEAR_MISSES["valuation"]  # 4/5 fields exact, format valid
        reward, metrics = score_through_env(env, near, PERFECT)
        assert reward == 0.0
        assert metrics["strict_field_score"] == pytest.approx(0.8)
        assert metrics["format_valid_json"] == 1.0
        # The loose diagnostics are tracked in strict mode too (the study recorded
        # both verdicts side by side) — and still contribute nothing to the reward.
        assert metrics["loose_field_score"] == 1.0
        assert metrics["valuation_loose"] == 1.0

    def test_scoring_real_dataset_row(self):
        """End-to-end: take a real generated row, answer it perfectly, score 1.0."""
        env = load_environment(num_examples=8, seed=42, strict=True)
        row = env.get_dataset()[0]
        reward, _ = score_through_env(env, row["answer"], row["answer"])
        assert reward == 1.0

    def test_loose_mode_reward_and_metrics(self):
        env = load_environment(num_examples=4, seed=42, strict=False)
        # Valuation off by ~4.2%: loose credits it, strict does not.
        near = json.dumps({**GOLD, "valuation": 125000000})
        reward, metrics = score_through_env(env, near, PERFECT)
        assert reward == 1.0, "all 5 fields pass the loose criterion"
        assert metrics["valuation_loose"] == 1.0
        assert metrics["valuation_strict"] == 0.0
        assert metrics["all5_exact"] == 0.0
        assert metrics["strict_field_score"] == pytest.approx(0.8)

    def test_loose_mode_graded_steps(self):
        env = load_environment(num_examples=4, seed=42, strict=False)
        # Wrong company entirely (fails loose too) + everything else exact -> 0.8.
        text = perturbed("company", "Completely Different Co")
        reward, metrics = score_through_env(env, text, PERFECT)
        assert reward == pytest.approx(0.8)
        assert metrics["company_loose"] == 0.0


# --------------------------------------------------------------------------- #
# Real study transcripts — the ported rubric must reproduce the original       #
# grader's recorded verdicts (from judge_taskB/strict-easy-seed0).             #
# --------------------------------------------------------------------------- #
REAL_TRANSCRIPTS = [
    {
        "name": "perfect-record",
        "gold": {"company": "Lumen Robotics", "round": "Pre-Seed", "raise": 40000000,
                 "valuation": 120000000, "founders": ["Elena Ruiz"]},
        "raw_output": "{\n  \"company\": \"Lumen Robotics\",\n  \"round\": \"Pre-Seed\",\n  "
                      "\"raise\": 40000000,\n  \"valuation\": 120000000,\n  \"founders\": "
                      "[\"Elena Ruiz\"]\n}",
        "strict": 1.0, "all5": 1.0, "format_valid": 1.0,
        "field_strict": {"company": 1.0, "round": 1.0, "raise": 1.0, "valuation": 1.0, "founders": 1.0},
        "field_loose": {"company": 1.0, "round": 1.0, "raise": 1.0, "valuation": 1.0, "founders": 1.0},
    },
    {
        "name": "valuation-near-miss-loose-passes",
        "gold": {"company": "Northwind", "round": "Series C", "raise": 15000000,
                 "valuation": 75000000, "founders": ["Priya Nair", "Rosa Diaz", "Omar Haddad"]},
        "raw_output": "{\n  \"company\": \"Northwind\",\n  \"round\": \"Series C\",\n  "
                      "\"raise\": 15000000,\n  \"valuation\": 71500000,\n  \"founders\": [\n    "
                      "\"Priya Nair\",\n    \"Rosa Diaz\",\n    \"Omar Haddad\"\n  ]\n}",
        "strict": 0.8, "all5": 0.0, "format_valid": 1.0,
        "field_strict": {"company": 1.0, "round": 1.0, "raise": 1.0, "valuation": 0.0, "founders": 1.0},
        "field_loose": {"company": 1.0, "round": 1.0, "raise": 1.0, "valuation": 1.0, "founders": 1.0},
    },
    {
        "name": "distractor-grabbed-round-and-raise",
        "gold": {"company": "Onyx", "round": "Series A", "raise": 3000000,
                 "valuation": 24000000, "founders": ["Hassan Ali"]},
        "raw_output": "{\n  \"company\": \"Onyx\",\n  \"round\": \"Series C\",\n  \"raise\": "
                      "2500000,\n  \"valuation\": 24000000,\n  \"founders\": [\"Hassan Ali\"]\n}",
        "strict": 0.6, "all5": 0.0, "format_valid": 1.0,
        "field_strict": {"company": 1.0, "round": 0.0, "raise": 0.0, "valuation": 1.0, "founders": 1.0},
        "field_loose": {"company": 1.0, "round": 0.0, "raise": 0.0, "valuation": 1.0, "founders": 1.0},
    },
]


class TestRealStudyTranscripts:
    @pytest.mark.parametrize("t", REAL_TRANSCRIPTS, ids=lambda t: t["name"])
    def test_direct_grader_reproduces_verdicts(self, t):
        text, gold = t["raw_output"], t["gold"]
        assert grade(text, gold) == pytest.approx(t["strict"])
        assert all_five_exact(text, gold) == t["all5"]
        assert format_valid(text) == t["format_valid"]
        obj = extract_json(text)
        assert per_field_strict_b(obj, gold) == t["field_strict"]
        assert per_field_loose_b(obj, gold) == t["field_loose"]

    @pytest.mark.parametrize("t", REAL_TRANSCRIPTS, ids=lambda t: t["name"])
    def test_env_scoring_path_reproduces_verdicts(self, t):
        env = load_environment(num_examples=4, seed=42, strict=True)
        reward, metrics = score_through_env(env, t["raw_output"], json.dumps(t["gold"]))
        assert reward == t["all5"]
        assert metrics["all5_exact"] == t["all5"]
        assert metrics["strict_field_score"] == pytest.approx(t["strict"])
        assert metrics["format_valid_json"] == t["format_valid"]
        for field in FIELDS:
            assert metrics[f"{field}_strict"] == t["field_strict"][field]
            assert metrics[f"{field}_loose"] == t["field_loose"][field]

    @pytest.mark.parametrize("t", REAL_TRANSCRIPTS, ids=lambda t: t["name"])
    def test_loose_env_reproduces_loose_verdicts(self, t):
        env = load_environment(num_examples=4, seed=42, strict=False)
        expected_loose = sum(t["field_loose"].values()) / 5.0
        reward, metrics = score_through_env(env, t["raw_output"], json.dumps(t["gold"]))
        assert reward == pytest.approx(expected_loose)
        for field in FIELDS:
            assert metrics[f"{field}_loose"] == t["field_loose"][field]
