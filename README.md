# research-study

A reproducible RLVR (Reinforcement Learning from Verifiable Rewards) study in which I GRPO+LoRA fine-tune `Qwen/Qwen2.5-1.5B-Instruct` on two startup-deal-finance tasks under a strict versus a loose reward grader, then measure whether the loose ("tolerant") reward induces grader-gaming and whether that gaming transfers across the two tasks.

## Research question

I want to know whether loosening the reward grader teaches the model to game the grader rather than do the task, and whether that behaviour is task-specific or transfers across tasks.

The single thing I vary deliberately is the **reward dial**:

- **strict** reward uses each task's exact-match grader.
- **loose** reward uses the same grader with pre-registered tolerances.

I train under both settings (plus two data-difficulty mixes and three seeds), then score every trained adapter on a sealed out-of-distribution (OOD) test set with *both* graders on the *same* saved generations.

**What "reward hacking" means here.** Because the loose grader is strictly more forgiving than the strict grader, for any single output `loose >= strict`. I operationalise reward hacking as the per-band **gap**:

- Task A: `gap_pp = loose_pct - strict_pct` per OOD band.
- Task B: `gap = loose_field - strict_field` per field, per band (never a single blended number).

A large gap means the model is producing outputs the loose grader accepts but the strict grader rejects — i.e. it is exploiting the tolerance instead of being correct. My pre-stated expectation is that the largest gaps appear in the **loose-reward** cells, concentrated on the **hard** OOD band. The cross-task step then asks whether the same loose-reward effect shows up on both tasks.

Note on terminology: there is **no LLM-as-judge** in this repo. "Judging" means re-running each frozen adapter to generate text and then scoring that text with deterministic, pure, rule-based graders. The `judge_config` is a frozen, hash-pinned identity bundle, and its `model_name` is the model being *scored*, not a separate judge model.

### The two tasks

- **Task A — "Deal math" (ownership %).** Given a messy sentence about a startup funding round (a raise and a pre-money valuation), the model computes the new investor's ownership percentage: `ownership% = raise / (pre_money + raise) * 100`, rounded to two decimals. The gold answer is always computed in code, never hand-typed.
  - *strict* grader: exact match after 2-dp `ROUND_HALF_UP`.
  - *loose* grader: within `±0.50` percentage points (inclusive).
- **Task B — deal extraction (prose → JSON).** Given prose describing a round, the model emits a JSON object with keys `company`, `round`, `raise`, `valuation`, `founders`. A structured record is sampled first and rendered into prose, so the record *is* the gold. Each item is scored as the fraction of the 5 fields that match.
  - *strict* grader: exact normalized-string / whole-dollar match; set-equal founders.
  - *loose* grader: fuzzy names (token-set ratio `>= 0.80`), canonicalized round labels (`"A" == "Series A"`, `"Pre-Seed" == "Preseed"`), and `±10%` numeric tolerance.

Both graders are deterministic and pure; both data generators are seed-deterministic.

### Experimental design

A 2×2×3 grid, run for each task:

- reward_mode ∈ {`strict`, `loose`} (the reward dial)
- difficulty ∈ {`easy`, `easy_hard`} (the training data mix)
- seed ∈ {`0`, `1`, `2`}

= **12 adapters per task** (24 total), plus one untrained baseline per task. The training data difficulty conditions (`easy` = easy-only; `easy_hard` = half easy + half hard) hold the total item count fixed at 256 — only the mix changes. These training conditions are deliberately distinct from the two **OOD evaluation bands** (`easy`, `hard`).

## Repository structure

```
research-study/
  study_config.py            Frozen single source of truth: MODEL_NAME, SYSTEM_PROMPT(_B),
                             generation kwargs, data sizes/seeds, LoRA + GRPO hyperparameters,
                             SMOKE overrides; snapshot() (excludes _B-suffixed keys).
  data/                      Output folder for generated datasets (gitignored; regenerated on demand).
                             Holds train_easy.jsonl, train_hard.jsonl, ood_test.jsonl (Task A)
                             and ood_test_b.jsonl (Task B).
  data_generation/
    generate.py              Task A generator with an argparse CLI. Builds train_easy / train_hard /
                             ood_test from one --seed; runs a no-leakage audit; disjoint train/OOD
                             template pools.
    generate_b.py            Task B generator. Library only (no CLI): exposes build_all_b().
                             Samples a 5-field record first, renders to prose, asserts recoverability
                             and train/OOD template disjointness.
  graders/
    grader.py                Task A strict + loose graders: extract_answer/extract_number, grade()
                             (exact 2-dp ROUND_HALF_UP), grade_loose() (±0.50), format_valid(), accuracy().
    grader_b.py              Task B strict + loose graders: extract_json(), per-field matchers,
                             grade()/grade_loose() (per-field fraction of 5), format_valid(),
                             all_five_exact().
  training/
    tasks.py                 Torch-free task router. task_spec('A'|'B') wires the data builder, graders,
                             extractor, system prompt, HF repo prefix (rlvr-taskA-/rlvr-taskB-), gold codec.
    train_grpo.py            GRPO + LoRA trainer (TRL GRPOTrainer + peft, use_vllm=False). Pure helpers
                             (reward func, trainable-param guard, adapter proof) + the GPU path train_one().
                             torch/transformers/trl/peft imports are deferred so the module imports without a GPU.
    run_real.py              Wrapper for the 12 real runs: run_cell / run_calibration / run_batch /
                             check_completion. Idempotent skip-if-repo-exists, NaN/OOM classification,
                             fixed-count control assert, incremental reward flushing. Defines the grid
                             (ALL_CONDITIONS, ALL_SEEDS) and repo_id().
    smoke_test.py            Throwaway dress rehearsal (run_smoke) at seed 99991 with cfg.SMOKE: proves
                             GRPO+LoRA trains, saves, and reloads (local + Hub) before spending GPU on the 12 runs.
  evaluation/
    run_baseline.py          Phase 3: untrained-baseline runner. Greedy + seeded Pass@k passes,
                             guardrails, per-band metrics. Also defines the shared generation helpers.
    judge_taskA.py           Phase 4: Task A judge. judge_all() scores the 12 Task A adapters on the
                             sealed OOD set; frozen-identity (SHA-256) asserts; per-band metrics with
                             gap_pp; cross-seed spread; no-peek aggregation.
    judge_taskB.py           Phase 5 (Part 3): Task B judge. judge_all_b(); headline = all-5-exact;
                             per-field diagnostic; two Pass@k variants; max_new_tokens = 384.
    pilot_taskB.py           Phase 5 (Part 1): Task B calibration pilot (CLI). No training, no adapters;
                             reads difficulty on the untrained model; sealed OOD never generated (n_ood=0).
    hacking_report.py        Phase 4 reward-hacking detector for Task A. Re-reads saved greedy transcripts
                             and re-scores strict+loose (never re-generates); loose-minus-strict gap per band.
    hacking_report_b.py      Phase 5 reward-hacking detector for Task B. Same idea, per field per band.
    analyze_cross_task.py    Phase 6: read-only cross-task analysis (CLI). Ingests judge results, reports
                             dial effects with signed deltas and NULL overlap flags, draws no verdict.
  judge_results/             Committed Task A judge outputs, one subfolder per cell + baseline (see Results).
  judge_taskB/               Committed Task B judge outputs, one subfolder per cell + baseline.
  results/                   Raw analysis dumps (analyze_cross_task_output.txt, hacking_report_b_output.txt).
  tests/                     pytest suite (test_graders, test_grader_b, test_data_generation, test_generate_b,
                             test_train, test_tasks, test_eval, test_judge, test_judge_b, test_real,
                             test_analyze_cross_task) pinning config/prompt freezing and the scoring contracts.
  PHASE4_JUDGE_PROTOCOL.md   Locked pre-registration of the Task A judging + reward-hacking measurement.
  PHASE5_JUDGE_PROTOCOL_B.md Locked pre-registration of the Task B judging + reward-hacking measurement.
  TASKB_PREREG.md            Locked Task B pre-registration: prompt + grader contract + Pass@k field criterion.
  PILOT_LOG.md               Pilot draft notes; verbatim source of the frozen Task A SYSTEM_PROMPT.
  pyproject.toml             Project metadata + pytest config (testpaths=tests, pythonpath=.).
  requirements.txt           Declared deps (pytest only; the GPU stack is not declared here).
```

## How it works

The pipeline runs in phases. The pure scoring/aggregation cores are torch-free and unit-tested on a laptop; the GPU phases target a Kaggle 16GB T4 (fp16; no bf16 on Turing), with default output directories under `/kaggle/working`.

1. **Generate data.** Task A is generated from one seed via a CLI; Task B is built as a library call. Both produce disjoint train/OOD template pools, and Task A runs a no-leakage audit so the answer never appears in the prompt.

   ```bash
   python -m data_generation.generate --seed 0 --n-easy 500 --n-hard 500 \
       --n-ood-easy 100 --n-ood-hard 100 --out-dir data
   ```
   ```python
   from data_generation.generate_b import build_all_b
   build_all_b(seed, n_easy, n_hard, n_ood_easy, n_ood_hard)   # returns train_easy/train_hard/ood_test
   ```

2. **Train.** Each run is a GRPO+LoRA fine-tune whose reward is the task's grader (`strict` → `grade`, `loose` → `grade_loose`); scoring is never reimplemented in the trainer. `train_one` is the one reusable run that every real and smoke run calls. I first run a throwaway smoke test, then a calibration run, then the full batch of 12.

   ```python
   from training.smoke_test import run_smoke;        run_smoke()              # dress rehearsal (seed 99991)
   from training.run_real import run_calibration;    run_calibration()        # real run #1: strict/easy/seed0
   from training.run_real import run_cell;           run_cell(reward_mode, difficulty, seed, task='A')
   from training.run_real import run_batch, ALL_CONDITIONS, ALL_SEEDS
   run_batch([(r, d, s) for (r, d) in ALL_CONDITIONS for s in ALL_SEEDS], max_hours=11.0, task='A')
   ```

   Each run saves only the small LoRA adapter and (idempotently) pushes it to a private Hugging Face repo named `{user}/rlvr-task{A|B}-{reward_mode}-{difficulty}-seed{seed}`, plus a `results.json` with the full reward history and a config snapshot.

3. **Baseline.** Before scoring adapters, I measure the untrained model on the same sealed OOD set (greedy pass + a seeded Pass@k pass) so I have a trained-vs-untrained reference.

   ```python
   run_baseline.run_baseline(model, tokenizer, out_dir='/kaggle/working/baseline', task='A')
   ```

4. **Judge.** Each judge re-generates greedy + Pass@k completions for all 12 adapters and scores them with the frozen graders. Before any scoring, the harness asserts the sealed OOD-set hash and the judge-config hash, proves each adapter actually loaded (non-zero LoRA weights + LoRA-shape read-back), and checks the loaded repo matches the cell. Runs are idempotent (skip if `results.json` exists), failures are isolated, and the cross-cell comparison table is withheld until all 12 runs are present (no peeking).

   ```python
   judge_taskA.judge_all(out_dir='/kaggle/working/judge',        user='zachmeister')   # Phase 4, Task A
   judge_taskB.judge_all_b(out_dir='/kaggle/working/judge_taskB', user='zachmeister')  # Phase 5, Task B
   ```

   (Task B calibration is a separate Phase-5 Part-1 pilot — `python evaluation/pilot_taskB.py --run` — which never generates the sealed OOD set.)

5. **Reward-hacking report.** Each detector re-reads the *saved* greedy transcripts and re-scores them with both graders (never re-generating, so strict and loose run on identical text), then reports the loose-minus-strict gap and dumps the worst-gap transcripts for human reading.

   ```python
   hacking_report.gap_report(out_dir='/kaggle/working/judge')               # Task A, per band
   hacking_report_b.gap_report_b(out_dir='/kaggle/working/judge_taskB')     # Task B, per field per band
   ```

6. **Cross-task analysis.** A read-only, laptop-only step ingests the Task A and/or Task B judge directories, compares dials (asserting exactly one dial differs, reporting a signed delta and a NULL flag when seed intervals overlap), and lines up headlines across tasks (Task A strict exact-match vs Task B all-5-exact). It draws no verdict.

   ```bash
   python evaluation/analyze_cross_task.py [out_dir_a] [out_dir_b]          # defaults to judge_results
   ```

## Running it

### Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell (use `source .venv/bin/activate` on macOS/Linux)
pip install -r requirements.txt
```

Requires Python `>= 3.9`. Note that `requirements.txt` and `pyproject.toml` declare only `pytest` — the GPU/RL stack (`torch`, `transformers`, `peft`, `trl`) that training and judging actually need is **not** declared here; it is supplied by the Kaggle GPU runtime where the heavy phases run.

### Tests

The full suite is pure Python and runs with no GPU/torch/Hub (model generation is stubbed):

```bash
pytest
```

### Key commands

```bash
# Generate Task A data (defaults shown)
python -m data_generation.generate --seed 0 --n-easy 500 --n-hard 500 \
    --n-ood-easy 100 --n-ood-hard 100 --out-dir data

# Task B calibration pilot
python evaluation/pilot_taskB.py --selftest          # CPU-only core checks
python evaluation/pilot_taskB.py --run               # Kaggle GPU pilot

# Cross-task analysis (read-only)
python evaluation/analyze_cross_task.py judge_results judge_taskB
```

Training and judging are invoked as Python functions (from a Kaggle notebook), not as shell CLIs — see the snippets under "How it works." Real runs require an `HF_TOKEN` with write scope, since Kaggle wipes local files at session end.

### Configuration

All hyperparameters and prompts live frozen in `study_config.py`:

- Model: `Qwen/Qwen2.5-1.5B-Instruct`, loaded fp16 (`USE_FP16=True`, `USE_BF16=False`).
- LoRA: `r=16`, `alpha=32`, `dropout=0.05`, target modules `q/k/v/o_proj` + `gate/up/down_proj`.
- GRPO: `NUM_GENERATIONS=4`, `LEARNING_RATE=1e-5`, `KL_BETA=0.04`, `MAX_PROMPT_LENGTH=512`, `MAX_COMPLETION_LENGTH=384` (Task B uses `MAX_COMPLETION_LENGTH_B=384`), `GRADIENT_ACCUMULATION_STEPS=6`, `GRAD_CLIP=1.0`, `TRAIN_EXAMPLES=256` (held fixed across `easy` vs `easy_hard`), `MAX_STEPS=200`.
- Data/eval: `N_EASY=500`, `N_HARD=500`, `N_OOD_EASY=100`, `N_OOD_HARD=100`, `DATA_SEED=0`, `EVAL_SEED=0`, `PASS_K=4`, `SAMPLE_TEMPERATURE=0.7`, `SAMPLE_TOP_P=0.95`, `MAX_NEW_TOKENS=768` (Task A decoding), `JUDGE_BATCH_SIZE=16` (pinned so Pass@k is reproducible).

## Results

Judged outputs are committed under `judge_results/` (Task A) and `judge_taskB/` (Task B). Each holds one subfolder per cell, named `{reward_mode}-{difficulty}-seed{seed}`, plus a re-measured untrained baseline in `baseline-untrained-seed-1`:

```
strict-easy-seed{0,1,2}        loose-easy-seed{0,1,2}
strict-easy_hard-seed{0,1,2}   loose-easy_hard-seed{0,1,2}
baseline-untrained-seed-1
```

Each cell folder contains exactly three files: `results.json` (per-band metrics + full provenance, including the adapter repo id, git commit, the sealed-OOD and judge-config SHA-256 hashes, grader version, sample seed, batch size, and the full config snapshot), `transcripts_greedy.jsonl`, and `transcripts_passk.jsonl`.

Metrics are always split into `easy` and `hard` bands, never blended:

- **Task A** (`judge_results/`): per band `n`, `strict_pct` (headline, greedy exact-match), `format_valid_pct`, `loose_pct`, `passk_pct`, `cut_off`, and `gap_pp` (= `loose_pct - strict_pct`).
- **Task B** (`judge_taskB/`): per band `n`, `all5_pct` (headline, all-5-exact), `strict_mean_pct` (per-field mean diagnostic), `format_valid_pct`, `passk_all5_pct`, `passk_field_pct`, `cut_off`, plus per-field dicts `field_strict_pct` / `field_loose_pct` / `field_gap_pp` over {`company`, `round`, `raise`, `valuation`, `founders`}.

`judge_results/` also contains a worst-gap transcript dump (`_worst_gap_loose_easy_hard.txt`). Raw analysis dumps live under `results/` (`analyze_cross_task_output.txt`, `hacking_report_b_output.txt`).

The cross-seed comparison table (4 cells × {easy, hard} bands, mean ± sample std over 3 seeds) is emitted only when all 12 runs are present. Final interpretation — which dial setting "won," and whether the effect transfers across tasks — is deferred to the Phase-6 cross-task step, which itself reports nulls and overlaps and names no winner.

## Published RL environments

Both tasks are packaged as self-contained verifiers environments (under `environments/`) and published on the Prime Intellect Environments Hub:

- **vc-deal-math** — Task A: messy funding-round prose → one computed ownership %.
  Hub: https://app.primeintellect.ai/dashboard/environments/zachspeck/vc-deal-math
  Install: `prime env install zachspeck/vc-deal-math`
- **vc-deal-extraction** — Task B: messy prose → a 5-field JSON record.
  Hub: https://app.primeintellect.ai/dashboard/environments/zachspeck/vc-deal-extraction
  Install: `prime env install zachspeck/vc-deal-extraction`

## Status / caveats

- **Scoring is rule-based, not an LLM judge.** Every metric comes from the deterministic graders in `graders/`; the "judge" only re-generates text and scores it. The graders are pure and unit-tested without a model.
- **Phase 4 (Task A judging)** is approved (2026-06-23) and implemented; its judge and reward-hacking harness exist. The protocol document records that, at the time it was frozen, nothing had yet run on a GPU.
- **Phase 5 (Task B judging)** protocol document was frozen in a pre-approval state: it records a pending PI sign-off on the Task B max-completion-length (`384`) and the Pass@k definition, and marks its judge-config hash as provisional. The Task B harness and judged outputs nonetheless now exist in the repo, so some "not yet implemented" / "nothing has run" language in the locked `.md` files reflects when each document was frozen and is out of date relative to the committed `judge_results/` and `judge_taskB/` outputs.
- **Dependency gap.** The torch/transformers/peft/trl stack is referenced only in code and config comments, not declared in `requirements.txt` or `pyproject.toml`; the heavy phases assume a Kaggle GPU environment that already provides it.
- **Task B data generation has no CLI.** `generate_b.py` exposes `build_all_b()` as a library function and does not itself write JSONL; the on-disk `data/ood_test_b.jsonl` is written by other code (not `generate_b.py`), so there is no self-contained shell command for Task B generation in this repo.
- **Pilot numbers are not baselines.** The Task A/Task B pilot figures in `PILOT_LOG.md` come from draft items, not the sealed OOD set, and are explicitly excluded as baselines; the trained-vs-untrained reference is the re-measured `baseline-untrained-seed-1` run.
- **Scope of claims.** Any findings are scoped to `Qwen2.5-1.5B-Instruct`, these two tasks, these dial settings, and these seeds — directional, not definitive. A null result or a reversal is a valid finding, not a failure.
