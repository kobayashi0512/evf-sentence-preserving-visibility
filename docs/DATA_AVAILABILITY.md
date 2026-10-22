# Data and model availability

The repository contains code, configuration files, aggregate manuscript-level results, and synthetic unit tests. It does not redistribute benchmark records, article full text, item-level predictions, or model weights.

## QASPER v0.3

The evaluation uses the official test/evaluator archive from the [QASPER project](https://github.com/allenai/qasper-led-baseline). The public archive URL is recorded in `scripts/fetch_public_assets.py`. Consult the upstream project and dataset documentation for the current data terms. The archive is downloaded locally and extracted under `data/qasper_test/`.

## Evidence Inference 2.0

The evaluation uses the v2.0 ranged release from the [Evidence Inference project](https://github.com/jayded/evidence-inference), retrieved from `https://evidence-inference.ebm-nlp.com/v2.0.tar.gz`. The expected SHA-256 is `6abe0d4ec0d331834981c0171c3c79d47515761867f82f1dc6066e43863a1586`. Preparation scripts extract the official validation or test cohort locally. Full-text articles are not redistributed; users should check article-specific PMC reuse terms before any further sharing.

The previously inspected test-article identifiers required to reconstruct the untouched test subset are provided in `configs/excluded_previously_inspected_test_pmcs.txt`. They are identifiers only, not article text or annotations.

## Model checkpoint

The study used the upstream zero-shot `BAAI/bge-reranker-v2-m3` checkpoint without fine-tuning. It is downloaded from the [official Hugging Face model page](https://huggingface.co/BAAI/bge-reranker-v2-m3) and is not stored in this repository. Consult the upstream model card and licence. No study-trained model weights were produced.
