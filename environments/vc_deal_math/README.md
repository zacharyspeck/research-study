# vc-deal-math

### Overview
- **Environment ID**: `vc-deal-math`
- **Short description**: VC deal math (RLVR study Task A): messy startup-funding prose in, one computed number out — the investor's ownership percentage — graded by the study's pre-registered strict/loose rule-based grader.
- **Tags**: math, word-problem, synthetic, single-turn, train, eval

### Task

Task A of a pre-registered RLVR sensitivity study (GRPO+LoRA on `Qwen/Qwen2.5-1.5B-Instruct`) asking which of two training levers moves a sealed out-of-distribution exam score more: the **reward dial** (loose vs. strict training grader) or the **task dial** (easy-only vs. easy+hard training mix, total example count held fixed at 256). The model sees one short passage of prose describing a startup funding round — a raise amount and a pre-money valuation — and must compute ONE number: the new investor's ownership percentage, `ownership% = raise / (pre_money + raise) * 100`, stated as `The answer is X` with X rounded to two decimals. The gold answer is always computed in code (never hand-typed) and rounded with 2-decimal `ROUND_HALF_UP`, matching the grader exactly. Half the items are **easy**: short single-sentence prompts with clean `$XM` figures, whose answers land in 5–55% on a one-decimal-clean (but never whole) percentage, so they measure the deal math rather than razor-precision rounding. Half are **hard**: deliberately ugly raise/pre-money values in mixed formats across items ("$2.5 million", "770,000", "$1.2B"), buried in longer, messier prose salted with 1–2 distractor numbers that are NOT part of the calculation (founding year, employee headcount, prior ARR, number of investor firms), with messy two-decimal answers in 8–55% that never land on a clean .00/.50. A no-leakage guard drops any item whose scenario text (the variable part of the prompt, before the fixed instruction) contains the answer. The study's sealed out-of-distribution exam templates are **excluded** from this package; only the train-template generator ships here.

- **Type**: single-turn (`SingleTurnEnv`) — one prompt in, one computed number out
- **Output format**: free-form working is allowed; the final answer should appear as `The answer is X` (X to two decimals). The system prompt is the study's frozen worked-example `SYSTEM_PROMPT` ("Show at most 2 short steps, then STOP..."). The grader extracts the number after the **last** answer cue ("answer is", "result =", "equals", ...), falling back to the last number in the text; it strips `%`/"percent", resolves thousands commas vs. decimal commas ("16,13" → 16.13), and treats unparseable output as 0.0 without raising.
- **Rubric overview**: the study's grader, ported verbatim — same extraction regexes, same `Decimal`-based 2-dp `ROUND_HALF_UP` on both sides of the comparison, same tolerances.

### Scoring

With `strict=True` (default), the scalar reward is the study's **strict grader**: `1.0` iff the extracted answer equals the gold ownership percentage exactly at two decimals (both sides rounded half-up via `Decimal`), else `0.0`. So against gold `16.13`: `16.13`, `16.130`, and `16.1349` (rounds to 16.13) score 1.0; `16.1` and `16.14` score 0.0.

With `strict=False`, the scalar reward is the study's **loose grader** (the reward dial): `1.0` iff the extracted answer is within `0.50` percentage points of gold, absolute and inclusive, else `0.0`. For gold `16.13` it accepts `15.63`–`16.63`. Both graders are binary and share the identical extraction/rounding pipeline; they differ only in the final comparison.

Both graders' verdicts are tracked as **weight-0.0 metrics in both modes** — the study read its per-item reward-hacking gap as `loose − strict` on the same generations — along with format validity. Only the weight-1.0 main reward differs between modes.

### Datasets
- **Primary dataset**: generated in-process, deterministically, by the study's Task-A train-template generator (ported from the study repo). No files are downloaded; no network access is needed.
- **Split sizes**: one train split of `num_examples` items (default 256, matching the study's training size), half easy + half hard templates, interleaved (easy, hard, easy, hard, …) so any prefix — e.g. a small `-n` eval — samples both bands. Same `(num_examples, seed)` ⇒ a byte-identical dataset (pinned by SHA-256 golden tests for seeds 0, 7, and 42).
- **Columns**: `question` (prose + instruction), `answer` (gold ownership % as a float, 2 dp), `info` (difficulty, raise, pre_money, post_money, exact unrounded ownership).

### Quickstart

Run an evaluation with default settings:

```bash
prime eval run vc-deal-math
```

Configure model and sampling:

```bash
prime eval run vc-deal-math \
  -m openai/gpt-4.1-mini \
  -n 20 -r 3 -t 768 -T 0.7 \
  -a '{"num_examples": 64, "seed": 42, "strict": true}'
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_examples` | int | `256` | Dataset size (half easy + half hard). The study trained on 256. |
| `seed` | int | `42` | Generator seed. Same args ⇒ identical dataset. |
| `strict` | bool | `true` | `true`: reward = the study's strict grader (exact 2-dp match). `false`: reward = the study's loose grader (within 0.50 pp, inclusive). |

**Required environment variables: none.** No network, no API keys, no data downloads — the dataset regenerates from the seed and scoring is pure Python.

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | The scalar reward: the strict verdict (`strict=true`) or the loose verdict (`strict=false`). |
| `strict_correct` | 1.0 iff the extracted answer equals gold exactly at 2 dp (round-half-up). |
| `loose_correct` | 1.0 iff the extracted answer is within 0.50 percentage points of gold (inclusive). |
| `format_valid_number` | 1.0 iff the output contains a parseable number at all (correct or not). |

### Provenance

Generator, grader, and system prompt are ported verbatim from the study repo (`data_generation/generate.py`, `graders/grader.py`, `study_config.py`); the grading protocol is pre-registered in the study's `PHASE4_JUDGE_PROTOCOL.md`. Tests pin the dataset to the study generator's exact output (golden rows + SHA-256 for seeds 0/7/42) and reproduce the original grader's stored verdicts on real judged transcripts from the study.
