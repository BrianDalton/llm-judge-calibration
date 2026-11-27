# Reproducing the experiments

See the [project report](README.md) for the methods and results.

## Screen a model’s expert ratings

From the repository root, run:

```bash
python3 scripts/screen_summeval_single_model.py --model M22 --output local_files/m22_screening.json
```

The script uses the bundled annotations, prints score distributions for coherence, fluency, consistency, and relevance, and saves a JSON report to `local_files/m22_screening.json`. It uses the Python standard library.

Data source: [SummEval](https://doi.org/10.1162/tacl_a_00373) ([original release](https://github.com/Yale-LILY/SummEval)). M22 summaries come from [BART](https://aclanthology.org/2020.acl-main.703/).

## Run tests

Activate the project virtual environment, then run these commands from the repository root:

```bash
python -m pip install -r requirements.txt
pytest
```

The tests use synthetic annotations and require no API keys or downloaded articles.

## Lint before committing

From the repository root, install the requirements and enable the Git hook once per clone:

```bash
.venv/bin/python -m pip install -r requirements.txt
git config --local core.hooksPath .githooks
```

The hook runs `ruff check .` on the working tree and blocks a commit if lint fails. It works from VS Code Source Control and the terminal. Install the Ruff extension in VS Code to see lint diagnostics while editing.

## Run the consistency judge

Prepare the paired article JSONL and benchmark split first. Install `requirements.txt` and set `OPENAI_API_KEY` in your terminal environment. From the repo root, start with a three-article calibration smoke test:

```bash
.venv/bin/python -m scripts.run_judge \
  --input local_files/model_annotations.aligned.paired.jsonl \
  --benchmark local_files/m20_consistency_benchmark.jsonl \
  --prompt prompts/consistency_v1.txt \
  --split cal --limit 3 \
  --output local_files/m20_consistency_v1_smoke.jsonl
```

For the full calibration run, omit `--limit` and use `local_files/m20_consistency_v1_cal.jsonl` as the output. Run the offline scorer on the complete calibration file:

```bash
.venv/bin/python -m scripts.score_judge \
  --benchmark local_files/m20_consistency_benchmark.jsonl \
  --predictions local_files/m20_consistency_v1_cal.jsonl \
  --split cal \
  --output local_files/m20_consistency_v1_cal_metrics.json
```

The judge defaults to `gpt-6.2-sol` and low reasoning effort. The runner saves usage and prompt metadata with every result. Use a new output path for each run. To validate the inputs without spending API credits, add `--dry-run`. The scorer requires a complete set of predictions for the selected split. After freezing the prompt and scoring rules, run and score `--split eval`. To compare a revision with v1, give the scorer the revision in `--predictions` and the v1 file in `--baseline`.

## Run the length bias experiment

The versioned input contains all 26 eligible original and padded summary pairs. Each pair contains an article ID, generator ID, original summary, and a longer summary that restates some of the same content. The runner uses v1's saved calibration predictions to check eligibility and reproduce its model settings.

```bash
.venv/bin/python -m scripts.run_length_bias \
  --input local_files/model_annotations.aligned.paired.jsonl \
  --benchmark local_files/m20_consistency_benchmark.jsonl \
  --pairs experiment_inputs/m20_length_bias_pairs.jsonl \
  --baseline local_files/m20_consistency_v1_cal.jsonl \
  --prompt prompts/consistency_v1.txt \
  --output local_files/m20_length_bias_all_scores.jsonl \
  --report local_files/m20_length_bias_all_results.json \
  --dry-run
```

Remove `--dry-run` to score all 26 pairs in 52 calls. Each summary version gets a new judge call. The report shows padded minus original scores, the word counts, and how often the freshly scored original reached 5.

## Run the judge stability test

The stability test scores each calibration summary in three fresh requests. It uses the v1 calibration file to recover the judge settings. That file’s scores are separate from the stability measurements. The runner requires zero cached input tokens for each response.

```bash
.venv/bin/python -m scripts.run_judge_stability \
  --input local_files/model_annotations.aligned.paired.jsonl \
  --benchmark local_files/m20_consistency_benchmark.jsonl \
  --baseline local_files/m20_consistency_v1_cal.jsonl \
  --prompt prompts/consistency_v1.txt \
  --output local_files/m20_judge_stability_scores.jsonl \
  --report local_files/m20_judge_stability_results.json
```

Add `--dry-run` to validate the inputs without calling the API. The complete calibration set requires 90 calls. The report shows each summary’s three scores and the gap between the highest and lowest.
