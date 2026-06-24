"""Pure-Python tests for the GRPO+LoRA training logic (no model, no GPU).

Importing training.train_grpo must not import torch/trl/peft (they are deferred),
so these run on a plain machine.
"""

import json

import pytest

import study_config as cfg
from study_config import SYSTEM_PROMPT, SYSTEM_PROMPT_B
from training.train_grpo import (
    TrainableParamError,
    adapter_is_loaded,
    build_reward_func,
    check_trainable_params,
    outputs_differ,
    probe_system_prompt,
    rewards_all_identical,
    task_completion_length,
    to_grpo_dataset,
)


def _chat(text):
    """A TRL conversational completion: a one-message assistant turn."""
    return [{"role": "assistant", "content": text}]


# --------------------------------------------------------------------------- #
# Task-B training-path wiring (what the smoke proves, unit-tested without a GPU) #
# --------------------------------------------------------------------------- #
def test_task_completion_length_taskB_uses_locked_override(monkeypatch):
    assert task_completion_length("A") == cfg.MAX_COMPLETION_LENGTH       # Task A unchanged
    assert task_completion_length("B") == cfg.MAX_COMPLETION_LENGTH_B == 384
    monkeypatch.setattr(cfg, "MAX_COMPLETION_LENGTH_B", None)             # falls back if unset
    assert task_completion_length("B") == cfg.MAX_COMPLETION_LENGTH
    assert task_completion_length("A") == cfg.MAX_COMPLETION_LENGTH       # A never affected


def test_probe_system_prompt_task_aware_default_unchanged():
    # default + explicit task='A' are byte-identical to the old hard-coded cfg.SYSTEM_PROMPT
    assert probe_system_prompt() == cfg.SYSTEM_PROMPT == SYSTEM_PROMPT
    assert probe_system_prompt("A") == SYSTEM_PROMPT
    # task='B' uses the JSON prompt -> a Task-B probe shows extraction, not Task-A math
    assert probe_system_prompt("B") == SYSTEM_PROMPT_B and probe_system_prompt("B") != SYSTEM_PROMPT
    # an explicit system_prompt overrides the task
    assert probe_system_prompt("B", system_prompt="custom probe") == "custom probe"


def test_taskB_reward_uses_grader_b_and_json_gold_roundtrip():
    gold = {"company": "Acme", "round": "Series A", "raise": 5000000,
            "valuation": 20000000, "founders": ["Jo Lee"]}
    rows = to_grpo_dataset([{"prompt": "Acme raised $5M ...", "answer": gold, "difficulty": "hard"}],
                           task="B")
    # SYSTEM_PROMPT_B is the system prompt (the JSON prompt, NOT Task A's math prompt)
    assert rows[0]["prompt"][0]["content"] == SYSTEM_PROMPT_B
    assert rows[0]["prompt"][0]["content"] != SYSTEM_PROMPT
    # dict gold is json.dumps'd into the dataset 'answer' column
    assert isinstance(rows[0]["answer"], str) and json.loads(rows[0]["answer"]) == gold
    # the reward fn json.loads's it back and grades per-field with grader_b -> reward VARIES
    rf = build_reward_func("strict", task="B")
    perfect, wrong = json.dumps(gold), json.dumps({**gold, "raise": 999})
    r = rf(completions=[perfect, wrong], answer=[rows[0]["answer"], rows[0]["answer"]])
    assert r[0] == 1.0 and r[1] == pytest.approx(0.8)   # 4/5 fields right -> not flat (round-trip OK)


# --------------------------------------------------------------------------- #
# Reward function — both completion formats, length, strict vs loose           #
# --------------------------------------------------------------------------- #
def test_reward_strict_plain_string_completions():
    rf = build_reward_func("strict")
    rewards = rf(completions=["The answer is 16.13", "The answer is 99"], answer=[16.13, 16.13])
    assert rewards == [1.0, 0.0]


def test_reward_strict_chat_list_completions():
    rf = build_reward_func("strict")
    comps = [_chat("Steps... The answer is 16.13"), _chat("I cannot tell")]
    rewards = rf(completions=comps, answer=[16.13, 16.13])
    assert rewards == [1.0, 0.0]


def test_reward_output_length_matches_completions():
    rf = build_reward_func("loose")
    comps = ["The answer is 20", "The answer is 30", "The answer is 40"]
    rewards = rf(completions=comps, answer=[20.0, 30.0, 40.0])
    assert len(rewards) == len(comps)


def test_reward_strict_vs_loose_on_near_miss():
    # gold 16.13, output 16.40: within 0.5 (loose passes), not exact (strict fails).
    strict = build_reward_func("strict")
    loose = build_reward_func("loose")
    comp = ["The answer is 16.40"]
    assert strict(completions=comp, answer=[16.13]) == [0.0]
    assert loose(completions=comp, answer=[16.13]) == [1.0]


def test_reward_fails_loudly_on_misaligned_lengths():
    rf = build_reward_func("strict")
    with pytest.raises(ValueError):
        rf(completions=["a", "b"], answer=[16.13])


def test_reward_fails_loudly_without_answer_column():
    rf = build_reward_func("strict")
    with pytest.raises(KeyError):
        rf(completions=["The answer is 16.13"])


def test_build_reward_func_rejects_bad_mode():
    with pytest.raises(ValueError):
        build_reward_func("fuzzy")


# --------------------------------------------------------------------------- #
# "rewards all identical" learning-signal check                               #
# --------------------------------------------------------------------------- #
def test_rewards_all_identical():
    assert rewards_all_identical([0.0, 0.0, 0.0]) is True
    assert rewards_all_identical([1.0, 1.0]) is True
    assert rewards_all_identical([0.0, 1.0, 0.0]) is False
    assert rewards_all_identical([]) is True


# --------------------------------------------------------------------------- #
# to_grpo_dataset shape                                                        #
# --------------------------------------------------------------------------- #
def test_to_grpo_dataset_shape():
    items = [{"prompt": "Compute the ownership.", "answer": 12.5, "difficulty": "easy"}]
    rows = to_grpo_dataset(items)
    assert len(rows) == 1
    row = rows[0]
    assert row["answer"] == 12.5
    assert row["prompt"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert row["prompt"][1] == {"role": "user", "content": "Compute the ownership."}


# --------------------------------------------------------------------------- #
# Trainable-parameter guard                                                    #
# --------------------------------------------------------------------------- #
def test_trainable_guard_zero_raises():
    with pytest.raises(TrainableParamError):
        check_trainable_params(0, 1_500_000_000)


def test_trainable_guard_full_model_raises():
    with pytest.raises(TrainableParamError):
        check_trainable_params(1_500_000_000, 1_500_000_000)


def test_trainable_guard_lora_fraction_passes():
    fraction = check_trainable_params(5_000_000, 1_500_000_000)
    assert 0.0 < fraction < 0.05


# --------------------------------------------------------------------------- #
# Adapter-load differ/same helper                                             #
# --------------------------------------------------------------------------- #
def test_outputs_differ():
    assert outputs_differ("identical", "identical") is False
    assert outputs_differ("  identical  ", "identical") is False   # whitespace-insensitive
    assert outputs_differ("base output", "trained output") is True


# --------------------------------------------------------------------------- #
# Adapter-loaded proof (LoRA weights present + non-zero)                       #
# --------------------------------------------------------------------------- #
def test_adapter_is_loaded_true_when_present_and_nonzero():
    assert adapter_is_loaded({"num_lora_params": 112, "num_nonzero_lora_params": 56}) is True


def test_adapter_is_loaded_false_when_missing():
    # No LoRA params at all -> it's the bare base model.
    assert adapter_is_loaded({"num_lora_params": 0, "num_nonzero_lora_params": 0}) is False


def test_adapter_is_loaded_false_when_all_zero():
    # LoRA params present but every one is zero -> not a usable adapter.
    assert adapter_is_loaded({"num_lora_params": 112, "num_nonzero_lora_params": 0}) is False
