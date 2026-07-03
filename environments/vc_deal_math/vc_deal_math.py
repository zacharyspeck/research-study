"""vc_deal_math — a self-contained verifiers environment for Task A of the
RLVR sensitivity study (VC deal math: messy funding-round prose -> one computed
number, the investor's ownership percentage).

One prompt in, one number out (SingleTurnEnv). The dataset is generated
deterministically in-process by the study's Task-A TRAIN-template generator
(ported from `data_generation/generate.py`); the sealed OOD exam templates are
EXCLUDED by design and do not appear anywhere in this package. Scoring is the
study's pre-registered grader (`graders/grader.py`), ported VERBATIM — same
answer extraction, same comma/percent normalization, same 2-decimal
ROUND_HALF_UP rounding, same tolerances.

Reward:
- strict=True  (default): the study's STRICT grader — 1.0 iff the extracted
  final-answer number equals the gold ownership % exactly at 2 decimals
  (round-half-up via Decimal), else 0.0.
- strict=False: the study's LOOSE grader — 1.0 iff the extracted number is
  within 0.50 percentage points of gold (absolute, inclusive), else 0.0.

The other grader's verdict and format-validity are attached as weight-0.0
metrics in both modes: tracked per rollout, never part of the scalar reward.

No environment variables and no network access are required; the dataset is
regenerated from the seed on every load.
"""

from __future__ import annotations

import random
import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

import verifiers as vf
from datasets import Dataset

# =========================================================================== #
# SECTION 1 — Grader, ported VERBATIM from graders/grader.py.                 #
# (Strict + loose graders for Task A deal-math ownership %. The match rule is #
# pre-registered: extract the final-answer number, round both sides to two    #
# decimals with ROUND_HALF_UP via Decimal, then exact equality — strict — or  #
# an absolute 0.50-percentage-point inclusive tolerance — loose. No parseable #
# number -> 0.0, never raising.)                                              #
# =========================================================================== #

# A numeric token: optional sign, digits with optional thousands commas, an
# optional decimal part -- or a bare ".5". A trailing sentence period is left
# out because the decimal part requires digits after the dot.
_NUMBER = r"[-+]?\d[\d,]*(?:\.\d+)?|[-+]?\.\d+"
_NUMBER_RE = re.compile(_NUMBER)

# "answer cue" + (optional linker) + (optional hedge) + the number it labels.
_ANSWER_CUE_RE = re.compile(
    r"(?:final\s+answer|answer|ownership(?:\s+percentage)?|result|equals?|=)"
    r"\s*(?:is|are|was|of|to|:|=)?"
    r"\s*(?:about|approximately|roughly|around|nearly|~)?"
    r"\s*(" + _NUMBER + r")",
    re.IGNORECASE,
)

_TWO_PLACES = Decimal("0.01")

# The loose grader's reward dial: an absolute tolerance in percentage points
# (inclusive). Decimal so the exact-0.50 boundary is not affected by float error.
LOOSE_TOLERANCE = Decimal("0.50")


def _normalize_token(token: str) -> str | None:
    """Turn a raw numeric token into a clean decimal string, or None.

    Resolves commas (thousands vs. decimal) and strips signs/punctuation. The
    result is parseable by both ``float`` and ``Decimal``.
    """
    s = token.strip().rstrip(".,")
    if not s:
        return None

    sign = ""
    if s[0] in "+-":
        sign = "-" if s[0] == "-" else ""
        s = s[1:]

    if "," in s and "." in s:
        # e.g. "1,234.56" -> American grouping; commas are thousands separators.
        s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        if (len(parts) == 2 and 1 <= len(parts[1]) <= 2
                and parts[0].isdigit() and parts[1].isdigit()):
            s = parts[0] + "." + parts[1]   # decimal comma, e.g. "16,13"
        else:
            s = s.replace(",", "")          # thousands separators

    candidate = sign + s
    try:
        float(candidate)
    except ValueError:
        return None
    return candidate


def _extract_number_str(text: str | None) -> str | None:
    """Select the model's final-answer number and return its normalized string."""
    if not text:
        return None

    cues = list(_ANSWER_CUE_RE.finditer(text))
    if cues:
        norm = _normalize_token(cues[-1].group(1))
        if norm is not None:
            return norm

    for token in reversed(_NUMBER_RE.findall(text)):
        norm = _normalize_token(token)
        if norm is not None:
            return norm
    return None


def _round2(value: float | str) -> Decimal | None:
    """Round to two decimals, half-up, via Decimal (no float drift)."""
    try:
        return Decimal(str(value)).quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        return None


def extract_answer(text: str | None) -> float | None:
    """Find the model's final-answer number; return it as a float, or None.

    This is the public "find the number in the model's output" step. It is the
    basis for the format-validity metric (:func:`format_valid`) and shares the
    same underlying extraction logic that :func:`grade` uses internally.
    Returns ``None`` when the text contains no parseable number.
    """
    norm = _extract_number_str(text)
    return float(norm) if norm is not None else None


# Backwards-compatible alias for the previous public name.
extract_number = extract_answer


def grade(model_output: str | None, gold: float) -> float:
    """Strict binary score: 1.0 if the model's answer matches gold, else 0.0.

    Parameters
    ----------
    model_output : the raw, free-form model output text.
    gold         : the gold ownership percentage (a float, e.g. 16.13).
    """
    extracted = _extract_number_str(model_output)
    if extracted is None:
        return 0.0
    e, g = _round2(extracted), _round2(gold)
    if e is None or g is None:
        return 0.0
    return 1.0 if e == g else 0.0


def grade_loose(output_text: str | None, gold: float) -> float:
    """Loose binary score (the reward dial): 1.0 if within LOOSE_TOLERANCE.

    Built on top of the strict grader: it uses the same :func:`extract_answer`
    and the same 2-decimal round-half-up via Decimal that :func:`grade` uses.
    The ONLY difference is the final comparison -- instead of requiring exact
    equality, it accepts any extracted value within ``LOOSE_TOLERANCE``
    percentage points of gold (absolute, inclusive). For gold 16.13 it accepts
    15.63 .. 16.63 inclusive.
    """
    extracted = extract_answer(output_text)
    if extracted is None:
        return 0.0
    e, g = _round2(extracted), _round2(gold)
    if e is None or g is None:
        return 0.0
    return 1.0 if abs(e - g) <= LOOSE_TOLERANCE else 0.0


def format_valid(text: str | None) -> float:
    """Format-validity metric: 1.0 if the model produced a number, else 0.0.

    One of the study's three metrics (correctness, format-validity, Pass@k).
    It asks only whether the model emitted a parseable number at all -- it does
    NOT check whether that number is correct.
    """
    return 1.0 if extract_answer(text) is not None else 0.0


def accuracy(model_outputs, golds) -> float:
    """Mean strict score over a batch of (output, gold) pairs."""
    outputs = list(model_outputs)
    gold_list = list(golds)
    if len(outputs) != len(gold_list):
        raise ValueError("model_outputs and golds must be the same length")
    if not outputs:
        return 0.0
    return sum(grade(o, g) for o, g in zip(outputs, gold_list)) / len(outputs)


# =========================================================================== #
# SECTION 2 — Data generator, ported from data_generation/generate.py.        #
# TRAIN templates only: the study's sealed OOD exam templates are deliberately#
# NOT included in this package. Each item is a messy sentence about a startup #
# funding round; the model must compute the new investor's ownership percent  #
# from the stated raise and pre-money valuation. The gold answer is ALWAYS    #
# computed in code (never hand-typed) and rounded to two decimals. A          #
# no-leakage guard drops any item whose scenario text contains the answer.    #
# =========================================================================== #

INSTRUCTION = (
    "What ownership percentage does the investor receive in this round? "
    "Think step by step, then give your final answer on its own line as "
    "'The answer is X' (for example: The answer is 16.13)."
)

# The study's frozen Task-A system prompt (study_config.SYSTEM_PROMPT, verbatim
# from PILOT_LOG.md) — identical in the study's training and judging.
SYSTEM_PROMPT_A = (
    "You solve a short math problem. Compute ownership% = raise / (pre_money + raise) * 100. "
    "Show at most 2 short steps, then STOP and write exactly: 'The answer is X' where X is the "
    "number rounded to 2 decimals. Example: 'Raise 5M, pre-money 20M. Post-money = 25M. "
    "5/25*100 = 20. The answer is 20.'"
)

# Entity pools (names, not phrasings).
COMPANIES = [
    "Northwind", "Brightloom", "Quanta Labs", "Verdant", "Helix BioSystems",
    "Cobalt", "Pinepoint", "Orbital Foods", "Meridian AI", "Saffron",
    "Tidewater", "Lumen Robotics", "Cartography", "Driftwood", "Ember Health",
]
INVESTORS = [
    "Acacia Capital", "Brookline Ventures", "Cedar Partners", "Delta Growth",
    "Evergreen Capital", "Fathom Ventures", "Granite Partners", "Harbor Capital",
    "Ironwood", "Juniper Equity",
]
SECTORS = [
    "fintech", "biotech", "logistics", "climate", "robotics",
    "consumer", "enterprise software", "healthcare", "developer tools",
]

# Train templates. Placeholders: {company} {investor} {raise_amt} {pre_money}
# and, for hard, the distractor slots {year} {headcount} {sector} {arr}
# {num_investors}.
TRAIN_EASY_TEMPLATES = [
    "{company} raised {raise_amt} at a {pre_money} pre-money valuation.",
    "{investor} invested {raise_amt} in {company} at a {pre_money} pre-money valuation.",
    "{company} closed a {raise_amt} round at a {pre_money} pre-money valuation.",
    "{company} took {raise_amt} in new funding on a {pre_money} pre-money valuation.",
    "{investor} led a {raise_amt} round in {company} at a {pre_money} pre-money valuation.",
    "{company} secured {raise_amt} at a pre-money valuation of {pre_money}.",
]

TRAIN_HARD_TEMPLATES = [
    "Founded in {year}, {company} has grown to {headcount} employees. In a tangled "
    "financing round, {investor} agreed to put in {raise_amt}, with the pre-money "
    "valuation set at {pre_money}.",
    "{company}, a {sector} business, spent weeks negotiating before {investor} "
    "committed {raise_amt}; the pre-money valuation landed at {pre_money}. The "
    "company reported {arr} in ARR last year.",
    "After courting {num_investors} different firms, {company} finally closed. "
    "{investor} wired {raise_amt} into the company, and the term sheet pinned the "
    "pre-money valuation at {pre_money}; headcount had reached {headcount} by then.",
    "{company} was started in {year} and now serves customers across the {sector} "
    "market. Its newest round saw {investor} provide {raise_amt} at a pre-money "
    "valuation of {pre_money}.",
    "In a deal that took months, {investor} backed {company} with {raise_amt}; the "
    "pre-money valuation was negotiated to {pre_money}. The startup employs about "
    "{headcount} people.",
    "{company} closed last fiscal year with {arr} in revenue. After protracted "
    "talks, {investor} injected {raise_amt} and the pre-money valuation was fixed "
    "at {pre_money}.",
]

# Easy band parameters (forwards: clean simple inputs -> usually non-round answer).
EASY_RAISES = [n * 500_000 for n in range(2, 20)]        # $1.0M .. $9.5M, $0.5M steps
EASY_PRES = [n * 1_000_000 for n in range(8, 50)]        # $8M .. $49M, $1M steps
EASY_MIN_OWNERSHIP, EASY_MAX_OWNERSHIP = 5.0, 55.0       # accept band (else resample)


def compute_ownership(raise_d: float, pre_d: float) -> float:
    """Investor ownership percent: raise / (pre_money + raise) * 100."""
    post = pre_d + raise_d
    return raise_d / post * 100.0


def round2_half_up(value: float) -> float:
    """Round to 2 decimals using half-up, matching the strict grader.

    The grader normalizes model outputs with
    ``Decimal(str(x)).quantize(Decimal("0.01"), ROUND_HALF_UP)``; gold answers
    are rounded the same way so boundary values (e.g. 15.625 -> 15.63,
    18.125 -> 18.13) agree instead of diverging under banker's rounding.
    """
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def fmt_easy(value_dollars: float) -> str:
    """Standard "$XM" formatting for the easy band (e.g. $20M, $2.5M)."""
    return f"${value_dollars / 1e6:g}M"


def format_money(value_dollars: float, style: str) -> str:
    """Render a dollar amount in a chosen (value-preserving) style."""
    v = value_dollars
    if style == "m_suffix":
        return f"${v / 1e6:g}M"
    if style == "m_word":
        return f"${v / 1e6:g} million"
    if style == "b_suffix":
        return f"${v / 1e9:g}B"
    if style == "b_word":
        return f"${v / 1e9:g} billion"
    if style == "k_suffix":
        return f"${v / 1e3:g}K"
    if style == "commas_dollar":
        return f"${int(round(v)):,}"
    if style == "commas_plain":
        return f"{int(round(v)):,}"
    raise ValueError(f"unknown money style: {style}")


def sample_hard_values(rng: random.Random):
    """Pick deliberately ugly raise / pre-money values (mixed magnitudes)."""
    scale = rng.choices(
        ["millions", "billions", "hundred_k"], weights=[6, 2, 2]
    )[0]
    if scale == "millions":
        raise_d = round(rng.randrange(11, 99) / 10.0 * 1e6)      # $1.1M-$9.8M
        pre_d = round(rng.randrange(15, 260) / 10.0 * 1e6)       # $1.5M-$25.9M
    elif scale == "billions":
        raise_d = round(rng.randrange(11, 99) / 10.0 * 1e9)      # $1.1B-$9.8B
        pre_d = round(rng.randrange(15, 400) / 10.0 * 1e9)       # $1.5B-$39.9B
    else:  # hundred_k
        raise_d = rng.randrange(15, 96) * 10_000                 # 150,000-950,000
        pre_d = rng.randrange(30, 260) * 10_000                  # 300,000-2,590,000
    return raise_d, pre_d


def styles_for(value_dollars: float) -> list[str]:
    """Display styles that render `value_dollars` naturally (and exactly)."""
    if value_dollars >= 1e9:
        return ["b_suffix", "b_word"]
    styles = ["commas_plain", "commas_dollar"]
    if value_dollars >= 1e6:
        styles += ["m_suffix", "m_word"]
    elif value_dollars >= 1e3 and value_dollars % 1000 == 0:
        styles += ["k_suffix"]
    return styles


def make_distractors(rng: random.Random) -> dict:
    """Numbers/words that are NOT part of the calculation."""
    arr_style = rng.choice(["m_suffix", "m_word", "commas_dollar"])
    arr_value = rng.choice([1e6, 2e6, 3e6, 4e6, 8e6, 12e6])
    return {
        "year": str(rng.randint(2005, 2021)),
        "headcount": str(rng.choice([12, 18, 25, 40, 60, 85, 120, 180, 250, 400])),
        "sector": rng.choice(SECTORS),
        "arr": format_money(arr_value, arr_style),
        "num_investors": str(rng.randint(2, 9)),
    }


def _item(prompt: str, answer: float, difficulty: str,
          raise_d: float, pre_d: float, ownership: float) -> dict:
    return {
        "prompt": prompt,
        "answer": answer,
        "difficulty": difficulty,
        "info": {
            "raise": float(raise_d),
            "pre_money": float(pre_d),
            "post_money": float(pre_d + raise_d),
            "ownership": float(ownership),
        },
    }


def make_easy_item(rng: random.Random, templates: list[str]):
    """One easy item (forwards; gold clean to 1 decimal place, not whole)."""
    while True:
        raise_d = rng.choice(EASY_RAISES)
        pre_d = rng.choice(EASY_PRES)
        ownership = compute_ownership(raise_d, pre_d)
        if not (EASY_MIN_OWNERSHIP <= ownership <= EASY_MAX_OWNERSHIP):
            continue
        answer = round2_half_up(ownership)
        cents = int(Decimal(str(answer)) * 100)
        if cents % 10 == 0 and cents % 100 != 0:   # 1-dp clean, not whole
            break

    template = rng.choice(templates)
    scenario = template.format(
        company=rng.choice(COMPANIES),
        investor=rng.choice(INVESTORS),
        raise_amt=fmt_easy(raise_d),
        pre_money=fmt_easy(pre_d),
    )
    prompt = scenario + " " + INSTRUCTION
    return _item(prompt, answer, "easy", raise_d, pre_d, ownership), scenario


def make_hard_item(rng: random.Random, templates: list[str]):
    """One hard item (ugly inputs -> messy mid-range decimal answer)."""
    while True:
        raise_d, pre_d = sample_hard_values(rng)
        ownership = compute_ownership(raise_d, pre_d)
        if not (8.0 <= ownership <= 55.0):
            continue
        if round(ownership * 100) % 50 == 0:   # reject clean .00 / .50
            continue
        break
    answer = round2_half_up(ownership)

    fill = make_distractors(rng)
    fill.update(
        company=rng.choice(COMPANIES),
        investor=rng.choice(INVESTORS),
        raise_amt=format_money(raise_d, rng.choice(styles_for(raise_d))),
        pre_money=format_money(pre_d, rng.choice(styles_for(pre_d))),
    )
    template = rng.choice(templates)
    scenario = template.format(**fill)
    prompt = scenario + " " + INSTRUCTION
    return _item(prompt, answer, "hard", raise_d, pre_d, ownership), scenario


# No-leakage logic.
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_OWNERSHIP_WORDS = ("ownership", "owns", "stake", "equity", "holds")


def scenario_of(prompt: str) -> str:
    """The variable part of the prompt (everything before the instruction)."""
    return prompt.split(INSTRUCTION)[0].strip()


def answer_in_scenario(scenario: str, answer: float) -> bool:
    """True if the scenario leaks the ownership answer.

    A leak is: a "%"/"percent"/ownership-word reference, or any complete numeric
    token whose value equals the gold answer (to 2 dp).
    """
    low = scenario.lower()
    if "%" in scenario or "percent" in low:
        return True
    if any(w in low for w in _OWNERSHIP_WORDS):
        return True
    for tok in _NUM_RE.findall(scenario):
        if round(float(tok.replace(",", "")), 2) == answer:
            return True
    return False


def answer_leaks(prompt: str, answer: float) -> bool:
    """Full-prompt leak check used by the audit."""
    # The answer must never appear as a percentage anywhere in the prompt.
    for s in {f"{answer:.2f}", f"{answer:g}", str(answer)}:
        for suffix in ("%", " %", " percent", "percent"):
            if s + suffix in prompt:
                return True
    # ...and must not appear in the (variable) scenario text at all.
    return answer_in_scenario(scenario_of(prompt), answer)


def audit_no_leak(items: list[dict]) -> tuple[int, int]:
    """Return (num_checked, num_leaks) across all items."""
    leaks = sum(1 for it in items if answer_leaks(it["prompt"], it["answer"]))
    return len(items), leaks


def generate_dataset(n: int, difficulty: str, templates: list[str],
                     rng: random.Random) -> list[dict]:
    """Build n leak-free items, preferring unique prompts."""
    maker = make_easy_item if difficulty == "easy" else make_hard_item
    items: list[dict] = []
    seen: set[str] = set()
    tries = 0
    budget = n * 200 + 1000
    while len(items) < n and tries < budget:
        tries += 1
        item, scenario = maker(rng, templates)
        if answer_in_scenario(scenario, item["answer"]):
            continue
        if item["prompt"] in seen:
            continue
        seen.add(item["prompt"])
        items.append(item)
    # Top up with (rare) duplicates if the unique space was exhausted.
    while len(items) < n:
        item, scenario = maker(rng, templates)
        if answer_in_scenario(scenario, item["answer"]):
            continue
        items.append(item)
    return items


def build_train_items(num_examples, seed):
    """Deterministically build `num_examples` train items: half easy + half hard.

    Mirrors the study's `easy_hard` training condition (which used an even 256) and
    `build_all`'s seed derivation: easy items from ``random.Random(seed + 1)`` over
    the easy TRAIN templates, hard items from ``random.Random(seed + 2)`` over the
    hard TRAIN templates. For odd `num_examples` the extra item is easy. Same args
    -> an identical item list (easy block first, hard block second).
    """
    n_hard = num_examples // 2
    n_easy = num_examples - n_hard
    train_easy = generate_dataset(n_easy, "easy", TRAIN_EASY_TEMPLATES, random.Random(seed + 1))
    train_hard = generate_dataset(n_hard, "hard", TRAIN_HARD_TEMPLATES, random.Random(seed + 2))
    return train_easy + train_hard


# =========================================================================== #
# SECTION 3 — verifiers wiring (built against verifiers 0.1.14).              #
# Reward functions use the installed Rubric calling convention:               #
# named args from score_objects — (parser, completion, answer, **kwargs).     #
# =========================================================================== #


def _completion_text(parser, completion):
    """Final assistant text of a rollout (chat message list or plain string)."""
    text = parser.parse_answer(completion)
    return text if isinstance(text, str) else ""


# --- main rewards ---------------------------------------------------------- #
def strict_correct(parser, completion, answer, **kwargs) -> float:
    """The study's STRICT grader: exact 2-dp (round-half-up) match, 0.0/1.0."""
    return grade(_completion_text(parser, completion), answer)


def loose_correct(parser, completion, answer, **kwargs) -> float:
    """The study's LOOSE grader: within 0.50 percentage points (inclusive), 0.0/1.0."""
    return grade_loose(_completion_text(parser, completion), answer)


# --- weight-0.0 metric ----------------------------------------------------- #
def format_valid_number(parser, completion, answer, **kwargs) -> float:
    """1.0 iff the output contains a parseable number (correct or not)."""
    return format_valid(_completion_text(parser, completion))


# --- environment loader ---------------------------------------------------- #
def load_environment(num_examples: int = 256, seed: int = 42, strict: bool = True) -> vf.Environment:
    """Build the Task-A deal-math environment.

    Args:
        num_examples: dataset size (half easy + half hard templates). The study
            trained on 256.
        seed: generator seed. Same (num_examples, seed) -> an identical dataset.
        strict: True -> reward is the study's strict grader (exact 2-dp match).
            False -> reward is the study's loose grader (within 0.50 pp).

    The scalar reward comes ONLY from the first rubric function (weight 1.0);
    every other function is a weight-0.0 tracked metric. Both graders' verdicts
    are tracked in both modes, so the study's loose-minus-strict reward-hacking
    gap can be read directly off the metrics.
    """
    if num_examples < 1:
        raise ValueError(f"num_examples must be >= 1, got {num_examples}")

    items = build_train_items(num_examples, seed)
    # Deterministic easy/hard interleave (e0, h0, e1, h1, ...): evaluation takes the
    # FIRST n rows unshuffled, so any prefix must sample both bands. The item
    # multiset is exactly build_train_items' output; only the row order changes.
    n_hard = num_examples // 2
    easy, hard = items[:num_examples - n_hard], items[num_examples - n_hard:]
    interleaved = []
    for i in range(len(easy)):
        interleaved.append(easy[i])
        if i < len(hard):
            interleaved.append(hard[i])
    rows = []
    for item in interleaved:
        rows.append({
            "question": item["prompt"],
            "answer": item["answer"],
            "info": {
                "difficulty": item["difficulty"],
                "raise": item["info"]["raise"],
                "pre_money": item["info"]["pre_money"],
                "post_money": item["info"]["post_money"],
                "ownership": item["info"]["ownership"],
            },
        })
    dataset = Dataset.from_list(rows)

    parser = vf.Parser()
    if strict:
        funcs = [strict_correct, loose_correct, format_valid_number]
    else:
        funcs = [loose_correct, strict_correct, format_valid_number]
    weights = [1.0] + [0.0] * (len(funcs) - 1)
    rubric = vf.Rubric(funcs=funcs, weights=weights, parser=parser)

    return vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=SYSTEM_PROMPT_A,
        parser=parser,
        rubric=rubric,
    )
