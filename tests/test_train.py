"""Pure-Python tests for the GRPO+LoRA training logic (no model, no GPU).

Importing training.train_grpo must not import torch/trl/peft (they are deferred),
so these run on a plain machine.
"""

import pytest

from study_config import SYSTEM_PROMPT
from training.train_grpo import (
    TrainableParamError,
    adapter_is_loaded,
    build_reward_func,
    check_trainable_params,
    outputs_differ,
    rewards_all_identical,
    to_grpo_dataset,
)


def _chat(text):
    """A TRL conversational completion: a one-message assistant turn."""
    return [{"role": "assistant", "content": text}]


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
