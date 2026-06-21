# FROZEN config — mirrors PILOT_LOG.md. Do not change after runs begin.
#
# Single source of truth imported by every run (this baseline now; training and
# judging later) so settings can never drift apart.

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Worked-example system prompt — copied VERBATIM from the ```text block in
# PILOT_LOG.md. Must stay byte-identical; tests/test_eval.py asserts it against
# the file. Do not retype or paraphrase.
SYSTEM_PROMPT = "You solve a short math problem. Compute ownership% = raise / (pre_money + raise) * 100. Show at most 2 short steps, then STOP and write exactly: 'The answer is X' where X is the number rounded to 2 decimals. Example: 'Raise 5M, pre-money 20M. Post-money = 25M. 5/25*100 = 20. The answer is 20.'"

# Generation budget.
MAX_NEW_TOKENS = 768

# Greedy decode — the correctness / format-validity pass.
GREEDY_GEN_KWARGS = {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS}

# Sampling decode — the Pass@k pass.
PASS_K = 4
SAMPLE_TEMPERATURE = 0.7
SAMPLE_TOP_P = 0.95
SAMPLE_GEN_KWARGS = {
    "do_sample": True,
    "temperature": SAMPLE_TEMPERATURE,
    "top_p": SAMPLE_TOP_P,
    "max_new_tokens": MAX_NEW_TOKENS,
}

# Data — frozen seed + sizes (must match how the JSONL data is generated).
DATA_SEED = 0
N_EASY = 500
N_HARD = 500
N_OOD_EASY = 100
N_OOD_HARD = 100

# Evaluation — seed set before the sampling pass so Pass@k is reproducible.
EVAL_SEED = 0


def snapshot() -> dict:
    """All frozen config constants, for printing on screen and saving with results."""
    return {
        name: value
        for name, value in globals().items()
        if name.isupper() and not name.startswith("_")
    }
