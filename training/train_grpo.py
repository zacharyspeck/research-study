"""GRPO + LoRA trainer for Task A (TRL's GRPOTrainer, single-GPU, in-process gen).

Tooling decision (approved by the PI): TRL `GRPOTrainer` + LoRA, `use_vllm=False`.
This changes none of the frozen experimental variables.

Design: the scoring / guard logic is kept pure and import-light so it is
unit-testable on a machine with no GPU. Every torch / transformers / trl / peft /
datasets import is deferred into the function that needs it, so importing this
module (and the tests) never requires those packages.
"""

from __future__ import annotations

import inspect
import math
import os
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import study_config as cfg
from data_generation.generate import build_all
from graders.grader import grade, grade_loose


class TrainableParamError(RuntimeError):
    """Raised when the LoRA setup would train nothing, or (nearly) the whole model."""


class NaNLossError(RuntimeError):
    """Raised when the loss goes NaN/inf (fp16 + RL instability) — a real finding."""


class AdapterProofError(RuntimeError):
    """Raised when a reloaded adapter does not change behavior vs. the base model."""


# --------------------------------------------------------------------------- #
# Pure, GPU-free logic (this is what tests/test_train.py exercises).           #
# --------------------------------------------------------------------------- #
def _completion_text(completion):
    """Extract assistant text from a TRL completion (chat message list OR string)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        assistant = [m.get("content", "") for m in completion
                     if isinstance(m, dict) and m.get("role") == "assistant"]
        if assistant:
            return "\n".join(assistant)
        return "\n".join(m.get("content", "") for m in completion if isinstance(m, dict))
    return str(completion)


def build_reward_func(reward_mode):
    """Return a TRL-compatible reward function backed by the real grader.

    TRL calls it as ``reward_func(completions=[...], answer=[...], prompts=[...], ...)``
    with one entry per completion. We never reimplement scoring — strict uses
    ``graders.grader.grade``; loose uses ``graders.grader.grade_loose``.
    """
    if reward_mode not in ("strict", "loose"):
        raise ValueError(f"reward_mode must be 'strict' or 'loose', got {reward_mode!r}")
    scorer = grade if reward_mode == "strict" else grade_loose

    def reward_func(completions, **kwargs):
        answers = kwargs.get("answer")
        if answers is None:
            raise KeyError(
                "reward function needs the dataset's 'answer' column passed as a kwarg"
            )
        if len(completions) != len(answers):
            raise ValueError(
                f"completions ({len(completions)}) and answer ({len(answers)}) are misaligned "
                f"— TRL must pass exactly one gold per completion."
            )
        return [float(scorer(_completion_text(c), g)) for c, g in zip(completions, answers)]

    reward_func.__name__ = f"reward_{reward_mode}"
    return reward_func


def to_grpo_dataset(items):
    """Map our items to TRL's conversational format (rows of message-list + answer).

    TRL applies the chat template internally, so we pass the message list (not a
    rendered string). The frozen SYSTEM_PROMPT is included on every row.
    """
    return [
        {
            "prompt": [
                {"role": "system", "content": cfg.SYSTEM_PROMPT},
                {"role": "user", "content": item["prompt"]},
            ],
            "answer": item["answer"],
        }
        for item in items
    ]


def check_trainable_params(trainable, total, max_fraction=0.05):
    """Hard-stop unless trainable params are > 0 and < max_fraction of total.

    Catches "training nothing" (LoRA not attached) and "training the whole model".
    Returns the trainable fraction on success.
    """
    if total <= 0:
        raise TrainableParamError(f"total parameter count must be positive, got {total}")
    fraction = trainable / total
    if trainable <= 0:
        raise TrainableParamError(
            "0 trainable parameters — LoRA is not attached, so you'd be training nothing."
        )
    if fraction >= max_fraction:
        raise TrainableParamError(
            f"{fraction:.2%} of parameters are trainable (>= {max_fraction:.0%}) — that looks "
            f"like full-model fine-tuning, not LoRA. Check the adapter setup."
        )
    return fraction


def rewards_all_identical(rewards):
    """True if every logged reward is the same value (no learning signal)."""
    rewards = list(rewards)
    if not rewards:
        return True
    return len(set(rewards)) == 1


def outputs_differ(base_output, adapter_output):
    """True if two generations differ (whitespace-insensitive)."""
    return (base_output or "").strip() != (adapter_output or "").strip()


def adapter_is_loaded(lora_stats):
    """Pure: True iff LoRA adapter weights are present AND at least one is non-zero.

    This is the real proof an adapter loaded (vs. silently being the bare base model):
    a freshly-loaded LoRA always has non-zero ``lora_A`` even before much training, so
    identical greedy text from a barely-trained adapter is fine — but missing or
    all-zero weights are not.
    """
    return (lora_stats.get("num_lora_params", 0) > 0
            and lora_stats.get("num_nonzero_lora_params", 0) > 0)


def _reward_from_logs(logs):
    """Pull a reward-mean value out of a TRL/Trainer log dict (version-tolerant)."""
    if "reward" in logs:
        return logs["reward"]
    for key, value in logs.items():
        if key.startswith("rewards/") and key.endswith("/mean"):
            return value
    return None


# --------------------------------------------------------------------------- #
# GPU path — torch / transformers / trl / peft imported lazily inside.         #
# Never touched by the unit tests.                                             #
# --------------------------------------------------------------------------- #
def _seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def make_peft_model(model_id=None, **lora_overrides):
    """Load the frozen base model in fp16 and attach LoRA; guard trainable %."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model_id = model_id or cfg.MODEL_NAME
    print(f"Loading {model_id} in fp16 ...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    lora = LoraConfig(
        r=lora_overrides.get("r", cfg.LORA_R),
        lora_alpha=lora_overrides.get("lora_alpha", cfg.LORA_ALPHA),
        lora_dropout=lora_overrides.get("lora_dropout", cfg.LORA_DROPOUT),
        target_modules=lora_overrides.get("target_modules", cfg.LORA_TARGET_MODULES),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()  # required for grad checkpointing + LoRA
    trainable, total = count_parameters(model)
    fraction = check_trainable_params(trainable, total)  # hard-stops on a bad setup
    print(f"trainable params: {trainable:,} / {total:,} ({fraction:.3%})")
    return model


def train_one(reward_mode, difficulty, seed, out_dir, save_repo=None, **hp):
    """One reusable GRPO+LoRA run (the 12 real runs call this with the 2x2 x 3 seeds).

    difficulty: "easy" -> easy-only; "easy_hard" -> half easy + half hard (the
    example COUNT is held fixed; only the difficulty mix changes).
    """
    import torch  # noqa: F401 — fail loudly here if run on a box with no torch
    from datasets import Dataset
    from transformers import AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    if difficulty not in ("easy", "easy_hard"):
        raise ValueError(f"difficulty must be 'easy' or 'easy_hard', got {difficulty!r}")

    # Resolve hyperparameters: cfg defaults, overridden by **hp (e.g. cfg.SMOKE).
    num_generations = hp.get("num_generations", cfg.NUM_GENERATIONS)
    max_steps = hp.get("max_steps", cfg.MAX_STEPS)
    max_prompt_length = hp.get("max_prompt_length", cfg.MAX_PROMPT_LENGTH)
    max_completion_length = hp.get("max_completion_length", cfg.MAX_COMPLETION_LENGTH)
    learning_rate = hp.get("learning_rate", cfg.LEARNING_RATE)
    kl_beta = hp.get("kl_beta", cfg.KL_BETA)
    grad_accum = hp.get("gradient_accumulation_steps", cfg.GRADIENT_ACCUMULATION_STEPS)
    logging_steps = hp.get("logging_steps", cfg.LOGGING_STEPS)
    n_examples = hp.get("train_examples", cfg.TRAIN_EXAMPLES)

    # Keep per-device batch divisible by num_generations (TRL's grouping rule).
    per_device_bs = num_generations
    if per_device_bs % num_generations != 0:
        raise ValueError("per_device_train_batch_size must be divisible by num_generations")

    print("=" * 72)
    print(f"train_one  reward={reward_mode}  difficulty={difficulty}  seed={seed}")
    print("=" * 72)
    _seed_everything(seed)
    print(f"seeded everything with {seed}")

    # Data — same builder/seed as the rest of the study.
    data = build_all(seed, cfg.N_EASY, cfg.N_HARD, cfg.N_OOD_EASY, cfg.N_OOD_HARD)
    if difficulty == "easy":
        items = data["train_easy"][:n_examples]
    else:
        half = n_examples // 2
        items = data["train_easy"][:half] + data["train_hard"][:half]
    train_dataset = Dataset.from_list(to_grpo_dataset(items))
    print(f"training on {len(train_dataset)} items ({difficulty})")

    model = make_peft_model()
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    reward_func = build_reward_func(reward_mode)

    # GRPOConfig arg names verified against the installed TRL (1.6.0): `max_prompt_length`
    # was REMOVED from GRPOConfig in TRL 1.x; `max_completion_length` and `beta` (KL) still
    # exist. To stay robust to whatever TRL version Kaggle has, build the full desired kwargs
    # from study_config and keep only the ones the installed GRPOConfig actually accepts
    # (introspected at runtime). Frozen values that a given TRL version doesn't expose are
    # simply not passed — never silently renamed.
    desired = dict(
        output_dir=str(out_dir),
        num_generations=num_generations,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        max_prompt_length=max_prompt_length,    # dropped automatically if unsupported
        max_completion_length=max_completion_length,
        learning_rate=learning_rate,
        beta=kl_beta,                           # KL coefficient
        max_steps=max_steps,
        logging_steps=logging_steps,
        gradient_checkpointing=cfg.GRAD_CHECKPOINTING,
        max_grad_norm=cfg.GRAD_CLIP,
        fp16=cfg.USE_FP16,
        bf16=cfg.USE_BF16,
        use_vllm=False,                         # in-process generation (single GPU)
        seed=seed,
        report_to="none",
        save_strategy="no",                     # we save only the LoRA adapter ourselves
    )
    supported = set(inspect.signature(GRPOConfig).parameters)
    dropped = sorted(k for k in desired if k not in supported)
    if dropped:
        print(f"note: installed GRPOConfig does not accept {dropped} — dropping them "
              f"(values stay frozen in study_config).")
    config = GRPOConfig(**{k: v for k, v in desired.items() if k in supported})

    reward_history = []

    class _RewardLogger(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if not logs:
                return
            loss = logs.get("loss")
            if loss is not None and (math.isnan(loss) or math.isinf(loss)):
                raise NaNLossError(
                    f"Loss became {loss} at step {state.global_step}. This is fp16 + RL "
                    f"instability, a real finding (not a silent failure). Stopping the run; "
                    f"lower the learning rate or revisit precision before continuing."
                )
            reward = _reward_from_logs(logs)
            if reward is not None:
                spread = logs.get("reward_std")
                reward_history.append(
                    {"step": state.global_step, "reward": reward, "reward_std": spread}
                )
                print(f"  step {state.global_step}: reward mean={reward:.4f} std={spread}")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_func],
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[_RewardLogger()],
    )
    trainer.train()

    # Learning-signal check.
    reward_values = [h["reward"] for h in reward_history]
    varied = not rewards_all_identical(reward_values)
    if varied:
        print("✅ reward varied across the run (there was a learning signal).")
    else:
        print("⚠️  WARNING: reward was flat (all-identical) across the run — NO learning signal. "
              "Check the grader / data / generation before trusting any real run.")

    # Save ONLY the LoRA adapter (small).
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    print(f"saved adapter ✅ → {out_dir}")

    pushed_repo = None
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if save_repo and token:
        model.push_to_hub(save_repo, private=True, token=token)
        pushed_repo = save_repo
        print(f"pushed adapter to PRIVATE HF repo ✅ → {save_repo}")
    elif save_repo and not token:
        print(f"⚠️  save_repo={save_repo!r} requested but no HF_TOKEN present — adapter NOT pushed. "
              f"A token is REQUIRED before the 12 runs (Kaggle wipes local files on session end).")

    return {
        "adapter_path": str(out_dir),
        "adapter_repo": pushed_repo,
        "reward_history": reward_history,
        "reward_varied": varied,
    }


def _greedy_generate(model, tokenizer, user_prompt, max_new_tokens=128):
    import torch
    messages = [
        {"role": "system", "content": cfg.SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc, do_sample=False, max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new = out[0, enc["input_ids"].shape[1]:]
    return tokenizer.decode(new, skip_special_tokens=True).strip()


def _lora_weight_stats(model):
    """Summarize a loaded peft model's LoRA weights (GPU helper; needs a real model)."""
    num_lora = 0
    num_nonzero = 0
    max_abs = 0.0
    for name, param in model.named_parameters():
        if "lora_" not in name:
            continue
        num_lora += 1
        peak = float(param.detach().abs().max())
        if peak > 0.0:
            num_nonzero += 1
        if peak > max_abs:
            max_abs = peak
    return {
        "num_lora_params": num_lora,
        "num_nonzero_lora_params": num_nonzero,
        "max_abs_lora": max_abs,
    }


def prove_adapter_loaded(base_model_id, adapter_src, probe_prompt, max_new_tokens=128):
    """Prove a trained adapter actually loaded.

    The REAL proof is that the LoRA adapter weights are present and non-zero after
    loading (so it isn't silently the bare base model). The base-vs-trained text
    comparison is printed for information only: a barely-trained smoke adapter can
    legitimately produce identical greedy text, so identical text is NOT a failure.
    We hard-fail only if the adapter weights are missing or all-zero.
    ``adapter_src`` may be a local path OR an HF repo id.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float16).to(device)
    base.eval()
    base_output = _greedy_generate(base, tokenizer, probe_prompt, max_new_tokens)
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    base2 = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float16).to(device)
    adapted = PeftModel.from_pretrained(base2, adapter_src).to(device)
    adapted.eval()
    adapter_output = _greedy_generate(adapted, tokenizer, probe_prompt, max_new_tokens)

    # Real proof: LoRA weights present and non-zero.
    lora_stats = _lora_weight_stats(adapted)
    loaded = adapter_is_loaded(lora_stats)

    # Informational: did the (possibly barely-trained) adapter change greedy text?
    differ = outputs_differ(base_output, adapter_output)
    print(f"[adapter proof] LoRA params: {lora_stats['num_lora_params']} "
          f"({lora_stats['num_nonzero_lora_params']} non-zero, "
          f"max|w|={lora_stats['max_abs_lora']:.3e})")
    print(f"[adapter proof] base    : {base_output[:200]}")
    print(f"[adapter proof] trained : {adapter_output[:200]}")
    print(f"[adapter proof] outputs differ: {differ}"
          + ("" if differ else "  (identical text is OK for a barely-trained smoke adapter)"))

    if not loaded:
        raise AdapterProofError(
            f"The reloaded adapter has no usable LoRA weights "
            f"({lora_stats['num_lora_params']} lora params, "
            f"{lora_stats['num_nonzero_lora_params']} non-zero) — it is missing or all-zero, so "
            f"the adapter did not actually load. (Silently the bare base model?)"
        )
    return {
        "base_output": base_output,
        "adapter_output": adapter_output,
        "differ": differ,
        "loaded": loaded,
        "lora_stats": lora_stats,
    }
