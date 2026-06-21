"""Task A -- "Deal math" data generator for an RLVR study.

Each item is a messy sentence about a startup funding round. The model must
compute the new investor's ownership percentage from the stated raise and
pre-money valuation:

    post_money       = pre_money + raise
    ownership_percent = raise / (pre_money + raise) * 100

The gold answer is ALWAYS computed in code (never hand-typed) and rounded to
two decimals.

Difficulty definitions
----------------------
easy : Built *forwards* from clean, simple inputs. We pick a clean raise
       ($1.0M-$9.5M in $0.5M steps) and a clean pre-money ($8M-$49M in $1M
       steps), compute ownership = raise / (pre + raise) * 100, and accept the
       candidate only if the gold (a) is in 5-55%, (b) is clean to ONE decimal
       place (its second decimal digit is 0, e.g. 12.5, 20.4), and (c) is NOT a
       whole number; otherwise resample. The answer is thus reachable without
       razor-precision rounding -- it measures the deal-math, not 2-dp rounding
       -- yet is still not a guessable whole number like {5,10,...,50}. Short,
       single-sentence, plain "$XM" phrasing, no distractor numbers, so it stays
       clearly easier than the hard band. Any candidate whose answer digits land
       in the prompt is still dropped by the no-leakage guard.

hard : Same calculation, but we pick deliberately ugly raise / pre-money values
       that produce messy decimal answers (e.g. 16.03, 24.01). The numbers are
       buried in longer, messier prose, written in mixed formats across items
       ("$2.5 million", "750,000", "$1.2B", "3,000,000"), with 1-2 distractor
       numbers that are NOT part of the calculation (founding year, employee
       headcount, prior revenue/ARR, number of investors). Answers are kept
       mid-range (8-55%) and never land on a clean .00 / .50.

OOD split
---------
Two completely disjoint pools of sentence templates are maintained: a TRAINING
pool and an OOD-TEST pool, sharing no templates or phrasings. We emit three
JSONL datasets in the output folder:

    train_easy.jsonl  -- easy items, TRAINING templates
    train_hard.jsonl  -- hard items, TRAINING templates
    ood_test.jsonl    -- a held-out test set drawn from OOD-TEST templates,
                         containing BOTH easy and hard items (tagged).

The OOD test set is never used for training; the same problem is therefore
worded differently at test time than during training.

Record schema (one JSON object per line)
----------------------------------------
    prompt     : str   -- the funding-round sentence + a fixed instruction.
                          Contains the inputs (raise, pre-money) but NEVER the
                          ownership percentage itself.
    answer     : float -- code-computed gold ownership %, rounded to 2 dp.
    difficulty : str   -- "easy" or "hard".
    info       : dict  -- {"raise", "pre_money", "post_money", "ownership"}
                          (ownership is the exact, unrounded value) for
                          debugging and leak-checking.

Everything is deterministic: one --seed controls all three datasets, so the
same seed reproduces byte-identical data. Files are plain JSONL, ready to load
later as a Hugging Face Dataset via
``datasets.load_dataset("json", data_files=...)``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

# --------------------------------------------------------------------------- #
# Fixed instruction (constant boilerplate appended to every scenario).         #
# It is identical for every item, so its example numbers ("20", "16.13")       #
# cannot serve as a per-item leak. The leak audit therefore scans the variable #
# scenario text (the part before this instruction).                            #
# --------------------------------------------------------------------------- #
INSTRUCTION = (
    "What ownership percentage does the investor receive in this round? "
    "Think step by step, then give your final answer on its own line as "
    "'The answer is X' (for example: The answer is 16.13)."
)

# --------------------------------------------------------------------------- #
# Entity pools (shared -- these are names, not phrasings, so they may appear   #
# in both train and OOD without breaking the template/phrasing disjointness).  #
# --------------------------------------------------------------------------- #
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

# --------------------------------------------------------------------------- #
# Template pools. TRAIN and OOD share NO templates and no distinctive          #
# phrasings (different verbs, sentence structure, and "pre-money" wording).    #
# Placeholders: {company} {investor} {raise_amt} {pre_money} and, for hard,    #
# the distractor slots {year} {headcount} {sector} {arr} {num_investors}.      #
# --------------------------------------------------------------------------- #
TRAIN_EASY_TEMPLATES = [
    "{company} raised {raise_amt} at a {pre_money} pre-money valuation.",
    "{investor} invested {raise_amt} in {company} at a {pre_money} pre-money valuation.",
    "{company} closed a {raise_amt} round at a {pre_money} pre-money valuation.",
    "{company} took {raise_amt} in new funding on a {pre_money} pre-money valuation.",
    "{investor} led a {raise_amt} round in {company} at a {pre_money} pre-money valuation.",
    "{company} secured {raise_amt} at a pre-money valuation of {pre_money}.",
]

OOD_EASY_TEMPLATES = [
    "When {company} was valued at {pre_money} pre-money, {investor} wrote a {raise_amt} check.",
    "{investor} contributed {raise_amt} to {company}, whose pre-money mark stood at {pre_money}.",
    "A {raise_amt} financing went into {company} against a pre-money figure of {pre_money}.",
    "{company} brought in {raise_amt}, having been pegged at {pre_money} pre-money.",
    "Backing {company} at {pre_money} pre-money, {investor} put up {raise_amt}.",
    "{company} added {raise_amt} in capital, with {pre_money} on the pre-money line.",
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

OOD_HARD_TEMPLATES = [
    "{company} dates back to {year}. In a convoluted transaction, an outside backer "
    "handed over {raise_amt}; negotiators fixed the pre-money figure at {pre_money}. "
    "Today the team numbers {headcount}.",
    "Operating in the {sector} space, {company} pulled together a complicated round: "
    "{investor} stumped up {raise_amt}, and the pre-money mark was settled at "
    "{pre_money}. The firm had logged {arr} in ARR.",
    "Some {num_investors} funds kicked the tires before the round came together. "
    "{investor} ultimately staked {raise_amt} on {company}, with the pre-money "
    "figure agreed at {pre_money}; staff had climbed to {headcount}.",
    "Established back in {year}, {company} works in {sector}. Its recent raise had "
    "{investor} supplying {raise_amt} against a pre-money figure of {pre_money}.",
    "Over a long negotiation, {investor} threw its weight behind {company} with "
    "{raise_amt}; the pre-money number was hammered out at {pre_money}. Around "
    "{headcount} people work there.",
    "{company} wrapped the prior year at {arr} in top-line revenue. After much "
    "back-and-forth, {investor} poured in {raise_amt} and the pre-money number came "
    "to {pre_money}.",
]

# Easy band parameters (forwards: clean simple inputs -> usually non-round answer).
EASY_RAISES = [n * 500_000 for n in range(2, 20)]        # $1.0M .. $9.5M, $0.5M steps
EASY_PRES = [n * 1_000_000 for n in range(8, 50)]        # $8M .. $49M, $1M steps
EASY_MIN_OWNERSHIP, EASY_MAX_OWNERSHIP = 5.0, 55.0       # accept band (else resample)


# --------------------------------------------------------------------------- #
# Core math + formatting helpers.                                              #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Distractor generators (hard band only).                                      #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Item construction.                                                           #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# No-leakage logic.                                                            #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Dataset assembly.                                                            #
# --------------------------------------------------------------------------- #
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


def build_all(seed: int, n_easy: int, n_hard: int,
              n_ood_easy: int, n_ood_hard: int) -> dict[str, list[dict]]:
    """Build all three datasets deterministically from a single seed."""
    rng_train_easy = random.Random(seed + 1)
    rng_train_hard = random.Random(seed + 2)
    rng_ood = random.Random(seed + 3)

    train_easy = generate_dataset(n_easy, "easy", TRAIN_EASY_TEMPLATES, rng_train_easy)
    train_hard = generate_dataset(n_hard, "hard", TRAIN_HARD_TEMPLATES, rng_train_hard)

    ood_easy = generate_dataset(n_ood_easy, "easy", OOD_EASY_TEMPLATES, rng_ood)
    ood_hard = generate_dataset(n_ood_hard, "hard", OOD_HARD_TEMPLATES, rng_ood)
    ood = ood_easy + ood_hard
    rng_ood.shuffle(ood)

    return {"train_easy": train_easy, "train_hard": train_hard, "ood_test": ood}


def write_jsonl(items: list[dict], path: Path) -> Path:
    """Write items as JSON Lines (one object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return path


# --------------------------------------------------------------------------- #
# CLI / demo.                                                                  #
# --------------------------------------------------------------------------- #
def _print_examples(items: list[dict], difficulty: str, k: int) -> None:
    shown = [it for it in items if it["difficulty"] == difficulty][:k]
    for i, it in enumerate(shown, 1):
        info = it["info"]
        print(f"  [{difficulty} #{i}] gold answer = {it['answer']}")
        print(f"    prompt: {it['prompt']}")
        print(
            f"    info:   raise=${info['raise']:,.0f}  pre=${info['pre_money']:,.0f}"
            f"  post=${info['post_money']:,.0f}  exact_ownership={info['ownership']:.4f}%"
        )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Task A 'Deal math' generator.")
    parser.add_argument("--seed", type=int, default=0,
                        help="single master seed (same seed -> identical data)")
    parser.add_argument("--n-easy", type=int, default=500,
                        help="number of easy TRAINING items")
    parser.add_argument("--n-hard", type=int, default=500,
                        help="number of hard TRAINING items")
    parser.add_argument("--n-ood-easy", type=int, default=100,
                        help="number of easy items in the OOD test set")
    parser.add_argument("--n-ood-hard", type=int, default=100,
                        help="number of hard items in the OOD test set")
    parser.add_argument("--out-dir", type=str, default="data",
                        help="output folder for the JSONL files")
    args = parser.parse_args()

    datasets = build_all(
        args.seed, args.n_easy, args.n_hard, args.n_ood_easy, args.n_ood_hard
    )

    out_dir = Path(args.out_dir)
    paths = {name: write_jsonl(items, out_dir / f"{name}.jsonl")
             for name, items in datasets.items()}

    all_items = [it for items in datasets.values() for it in items]
    checked, leaks = audit_no_leak(all_items)

    print("=" * 70)
    print("Task A 'Deal math' -- generation complete")
    print("=" * 70)
    for name, items in datasets.items():
        n_e = sum(1 for it in items if it["difficulty"] == "easy")
        n_h = sum(1 for it in items if it["difficulty"] == "hard")
        print(f"  {paths[name]}  ({len(items)} items: {n_e} easy, {n_h} hard)")
    print()

    status = "PASSED" if leaks == 0 else f"FAILED ({leaks} leaks)"
    print(f"No-leakage check: {status} "
          f"-- gold answer absent from all {checked} prompts.")
    print()

    print("-" * 70)
    print("3 EASY examples (training pool):")
    print("-" * 70)
    _print_examples(datasets["train_easy"], "easy", 3)

    print("-" * 70)
    print("3 HARD examples (training pool):")
    print("-" * 70)
    _print_examples(datasets["train_hard"], "hard", 3)

    print("-" * 70)
    print("OOD test examples (held out, disjoint templates):")
    print("-" * 70)
    _print_examples(datasets["ood_test"], "easy", 1)
    _print_examples(datasets["ood_test"], "hard", 1)


if __name__ == "__main__":
    main()
