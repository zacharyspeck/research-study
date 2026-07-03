# vc-deal-extraction

### Overview
- **Environment ID**: `vc-deal-extraction`
- **Short description**: VC deal extraction (RLVR study Task B): messy startup-funding prose in, a 5-field JSON record out, graded by the study's pre-registered strict/loose rule-based grader.
- **Tags**: extraction, json, synthetic, single-turn, train, eval

### Task

Task B of a pre-registered RLVR sensitivity study (GRPO+LoRA on `Qwen/Qwen2.5-1.5B-Instruct`) asking which of two training levers moves a sealed out-of-distribution exam score more: the **reward dial** (loose vs. strict training grader) or the **task dial** (easy-only vs. easy+hard training mix, total example count held fixed at 256). The model sees one short passage of messy prose describing a startup funding round — company, round name, amount raised, pre-money valuation, and founders, rendered in varied money formats ("$5M", "5 million", "$5,000,000") — and must output ONLY a JSON object with the keys `company`, `round`, `raise`, `valuation`, `founders`. A structured gold record is sampled from diverse pools *first* and then rendered into prose, so the gold is code-computed, never hand-labeled, and every field is recoverable from the text by construction. Half the items are **easy** (plain statements); half are **hard**, salted with distractors that bite — a prior-round raise amount in parallel phrasing, an advisor name in a founder-adjacent clause, and a competing post-round valuation — while the prose always disambiguates the true answer (temporal cues, an explicit "pre-money" tag, "founded by"). The study's sealed out-of-distribution exam templates are **excluded** from this package; only the train-template generator ships here.

- **Type**: single-turn (`SingleTurnEnv`) — one prompt in, one JSON answer out
- **Output format**: a JSON object with keys `company` (str), `round` (str), `raise` (int dollars), `valuation` (int pre-money dollars), `founders` (list of str). The system prompt is the study's frozen `SYSTEM_PROMPT_B` (worked-example, "no prose and no code fence").
- **Rubric overview**: the study's grader, ported verbatim. The grader parses the first JSON object out of the completion (tolerating code fences and surrounding prose), normalizes strings (lowercase, collapsed whitespace, stripped surrounding punctuation), parses numbers from `$12M` / `12 million` / `12,000,000`, and compares founders order-insensitively — correct parsing, not lenience. Unparseable output scores 0.0 and never raises.

### Scoring

With `strict=True` (default), the scalar reward is **all-5-exact**: `1.0` iff every one of the 5 fields matches the strict criterion (normalized string equality for `company`/`round`, whole-dollar numeric equality for `raise`/`valuation`, set-equality of normalized names for `founders`), else `0.0`. This is the study's headline record-correctness metric.

With `strict=False`, the scalar reward is the study's **loose per-field score**: the fraction of the 5 fields matching the pre-registered loose tolerances (fuzzy names at token-set ratio ≥ 0.80, round-label synonyms like "A" == "Series A", numbers within ±10%), i.e. values in {0.0, 0.2, ..., 1.0}. This is exactly the `grade_loose` reward dial the study trained against; no binary loose variant exists in the study, so none is invented here.

Everything else is a **weight-0.0 metric** — tracked per rollout in BOTH modes, never part of the reward: the per-field strict accuracies (`company_strict`, `round_strict`, `raise_strict`, `valuation_strict`, `founders_strict`), the per-field loose accuracies (`company_loose`, …, `founders_loose`), both aggregate scores (`strict_field_score`, `loose_field_score`, plus `all5_exact` in loose mode), and format validity (`format_valid_json`). The study recorded strict and loose verdicts side by side on the same generations, so the per-field loose-minus-strict reward-hacking gap can be read directly off the metrics.

### Datasets
- **Primary dataset**: generated in-process, deterministically, by the study's Task-B train-template generator (ported from the study repo). No files are downloaded; no network access is needed.
- **Split sizes**: one train split of `num_examples` items (default 256, matching the study's training size), half easy + half hard templates, interleaved (easy, hard, easy, hard, …) so any prefix — e.g. a small `-n` eval — samples both bands. Same `(num_examples, seed)` ⇒ a byte-identical dataset (pinned by a SHA-256 golden test).
- **Columns**: `question` (prose + extraction instruction), `answer` (the gold record as a JSON string), `info` (difficulty, source prose, rendered money strings, distractors).

### Quickstart

Run an evaluation with default settings:

```bash
prime eval run vc-deal-extraction
```

Configure model and sampling:

```bash
prime eval run vc-deal-extraction \
  -m openai/gpt-4.1-mini \
  -n 20 -r 3 -t 512 -T 0.7 \
  -a '{"num_examples": 64, "seed": 42, "strict": true}'
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_examples` | int | `256` | Dataset size (half easy + half hard). The study trained on 256. |
| `seed` | int | `42` | Generator seed. Same args ⇒ identical dataset. |
| `strict` | bool | `true` | `true`: reward = binary all-5-exact (strict grader). `false`: reward = the study's loose per-field score. |

**Required environment variables: none.** No network, no API keys, no data downloads — the dataset regenerates from the seed and scoring is pure Python.

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | The scalar reward: binary all-5-exact (`strict=true`) or the loose per-field score (`strict=false`). |
| `all5_exact` | 1.0 iff all 5 fields match the strict criterion (the study's headline). |
| `strict_field_score` | Fraction of the 5 fields matching strictly (0.0–1.0 in steps of 0.2). |
| `loose_field_score` | Fraction of the 5 fields within the loose tolerances (loose mode's reward). |
| `company_strict` … `founders_strict` | Per-field strict accuracy (weight 0.0, always tracked). |
| `company_loose` … `founders_loose` | Per-field loose accuracy (weight 0.0, always tracked). |
| `format_valid_json` | 1.0 iff the output parses to a JSON object with all five expected keys. |

### Provenance

Generator, grader, per-field metrics, and system prompt are ported verbatim from the study repo (`data_generation/generate_b.py`, `graders/grader_b.py`, `evaluation/judge_taskB.py`, `study_config.py`); the grader contract is pre-registered in the study's `TASKB_PREREG.md`. Tests reproduce the original grader's verdicts on real judged transcripts from the study.
