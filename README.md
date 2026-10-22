# Evidence Visibility under Fixed Display Budgets

This repository contains the code and publication-level aggregate results for the study **Evidence Visibility under Fixed Display Budgets: A Length-Conditioned Trade-Off in Segmentation and Packing**.

The study asks a narrow question: when a reader-facing view has a fixed maximum text budget, does changing selectable units from intact paragraphs to source-linked, sentence-preserving chunks change how much human-annotated evidence is visible? The evaluation uses known documents: retrieval of documents from a corpus is outside scope. The endpoint is overlap with annotated source text, not answer correctness, evidence sufficiency, reading comprehension, or clinical decision quality.

## Main results

At the 240-token display cap, sentence-preserving chunking increased mean source-character recall in both held-out evaluations for BGE. The effect was larger in Evidence Inference, and BM25 showed a similar gain there. The length-stratified results show a trade-off: chunking exposes some evidence in paragraphs that cannot fit intact, while it can reduce visibility for evidence already in shorter paragraphs.

| Dataset | Selector | Paragraph recall | Chunk recall | Chunk − paragraph (95% CI) |
|---|---|---:|---:|---:|
| QASPER test | BM25 | 0.243 | 0.251 | +0.76 pp (−0.36, +1.88) |
| QASPER test | BGE | 0.360 | 0.391 | +3.15 pp (+1.86, +4.48) |
| Evidence Inference test | BM25 | 0.129 | 0.267 | +13.82 pp (+9.76, +17.96) |
| Evidence Inference test | BGE | 0.187 | 0.324 | +13.70 pp (+9.03, +18.30) |

The samples were 1,164 QASPER questions from 396 papers and 953 Evidence Inference prompts from 239 previously uninspected articles. These are separate benchmark-specific estimates and are not pooled.

The primary comparison is a policy-level contrast: the same maximum display allowance is enforced, but whole-unit packing can leave different amounts unused. The results therefore do not isolate sentence boundaries from realized budget utilization. See [`results/RESULTS.md`](results/RESULTS.md) for budget sensitivity and the length-stratified trade-off.

## Repository contents

- `src/sentence_chunking.py`: deterministic sentence-preserving unit construction with source-character offsets.
- `scripts/`: frozen QASPER and Evidence Inference evaluation runners, data preparation, and supporting audits.
- `configs/`: protocols frozen before the corresponding test evaluations.
- `results/`: aggregate tables transcribed from the manuscript; no item-level predictions or source text are redistributed.
- `tests/`: synthetic unit tests for segmentation and offset behavior.
- `docs/`: data access, licensing, and reproduction notes.

The project did not train or fine-tune a model and therefore has no study-trained checkpoint to release. Experiments use the public `BAAI/bge-reranker-v2-m3` checkpoint; its weights are downloaded from the upstream model page and are not included here. `configs/model_weights_manifest.json` records the local experiment fingerprint.

## Quick start

Use Python 3.10 or 3.11. Install the dependencies and test the sentence chunker:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Download the public assets required for a full evaluation:

```bash
python scripts/fetch_public_assets.py --include qasper_test evidence_inference bge
```

This can download large datasets and model files into ignored `data/` and `models/` directories. Review the original provider terms before use. For exact dataset preparation and evaluation commands, follow [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Data and model sources

- **QASPER v0.3**: [official project](https://github.com/allenai/qasper-led-baseline); the test/evaluator archive is retrieved from the project’s public S3 release. The corpus is not copied into this repository.
- **Evidence Inference 2.0**: [official project](https://github.com/jayded/evidence-inference); the v2.0 ranged release is retrieved from the project’s public download endpoint. Article text and annotations are not copied into this repository.
- **BGE reranker**: [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3). No weights are hosted here.

The evaluation data and checkpoint remain subject to the terms of their respective providers. The code and original documentation in this repository are under the MIT License; that license does not relicense third-party datasets, articles, or model files.

## Citation

If you use the code or results, please cite the paper (bibliographic details will be added when available) and cite the original QASPER, Evidence Inference, and BGE resources. The repository is versioned; the paper points to release tag `v1.0.0` for the materials corresponding to this manuscript version.

## Scope

This is an empirical audit of evidence visibility in selected views from already identified documents. It does not establish that a selected view is sufficient, that a reader notices or understands all visible evidence, or that the approach improves scientific or clinical decisions.
