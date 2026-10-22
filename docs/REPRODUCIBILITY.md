# Reproduction guide

## 1. Environment and smoke test

Create a Python 3.10 or 3.11 environment, install `requirements.txt`, then run `python -m unittest discover -s tests -v`. This test path exercises sentence splitting and source-offset invariants using synthetic text; it does not download datasets or load the BGE model.

The experiments used PyTorch, Transformers, NumPy, and PySBD 0.3.4. The model tokenizer is the fast tokenizer shipped with `BAAI/bge-reranker-v2-m3`; unit lengths exclude query and special tokens. BGE scoring uses a maximum pair length of 512. Both device and package versions can affect runtimes; record the actual versions and device when reproducing.

## 2. Download assets

```bash
python scripts/fetch_public_assets.py --include qasper_test evidence_inference bge
```

The QASPER test archive is extracted under `data/qasper_test/`. The Evidence Inference archive remains under `data/archives/` because the preparation scripts read its official member structure directly. The BGE checkpoint is downloaded to `models/bge-reranker-v2-m3/`. None of these paths are tracked by Git.

## 3. Prepare Evidence Inference cohorts

Extract the official validation cohort:

```bash
python scripts/prepare_evidence_inference_official_dev.py \
  --archive data/archives/evidence-inference-v2.0-ranged.tar.gz \
  --out data/evidence_inference_validation
```

Prepare the untouched official-test article subset. The preparation excludes the previously examined article IDs in the repository manifest and writes the selected articles locally:

```bash
python scripts/prepare_evidence_inference_untouched_test.py \
  --archive data/archives/evidence-inference-v2.0-ranged.tar.gz \
  --previously-inspected-ids configs/excluded_previously_inspected_test_pmcs.txt \
  --out data/evidence_inference_untouched_test
```

## 4. Run the main evaluations

Locate the QASPER test JSON in the extracted archive, then run the frozen test analysis:

```bash
python scripts/run_qasper_test_chunk_confirmation.py \
  --test-json data/qasper_test/qasper-test-v0.3.json \
  --bge-model models/bge-reranker-v2-m3 \
  --out outputs/qasper_test_chunk_confirmation \
  --budgets 240 544 --max-unit-tokens 240 --max-length 512 \
  --bootstrap-reps 20000 --seed 20261016
```

Run the Evidence Inference validation cohort or untouched-test cohort by changing `--data` and `--out`:

```bash
python scripts/run_evidence_inference_official_dev_confirmation.py \
  --data data/evidence_inference_untouched_test \
  --bge-model models/bge-reranker-v2-m3 \
  --out outputs/evidence_inference_untouched_test \
  --budgets 240 544 --max-unit-tokens 240 --max-length 512 \
  --bootstrap-reps 20000 --seed 20261016
```

Use `--data data/evidence_inference_validation` for the official validation cohort. On the validation cohort, run the prespecified budget sweep described in `configs/budget_sensitivity.json` by passing the documented display budgets to the same runner. Do not select a budget or segmentation rule based on the untouched test results.

The runners write detailed row-level outputs locally. They are ignored by Git because they can contain benchmark identifiers and derived per-item data. Only aggregate tables are included in `results/`.

## 5. Statistical protocol and interpretation

The primary estimate is the paired chunk-minus-paragraph difference in source-character recall at 240 displayed tokens. Confidence intervals use 20,000 paired percentile bootstrap resamples clustered by source paper for QASPER or source article for Evidence Inference. The same query, selector, and whole-unit packing rule are used for both unitizations. The maximum displayed-text allowance is equal; realized token usage is not forced to be identical.

See the frozen JSON protocols in `configs/` for cohort construction, exclusions, outcomes, and interpretation boundaries. Do not pool QASPER and Evidence Inference metrics. Visibility of annotated text is not semantic evidence sufficiency, answer correctness, reader comprehension, or clinical utility.
