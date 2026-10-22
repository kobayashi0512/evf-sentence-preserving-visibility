#!/usr/bin/env python3
"""Engineering QA for sentence-preserving chunks; does not read evidence labels."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sentence_chunking import sentence_preserving_chunks


BLOCK_RE = re.compile(r"\S.*?(?=\n\s*\n|\Z)", re.S)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles", type=Path, default=Path("data/evidence_inference/articles"))
    parser.add_argument("--tokenizer", type=Path, default=Path("models/bge-reranker-v2-m3"))
    parser.add_argument("--max-unit-tokens", type=int, default=240)
    parser.add_argument("--out", type=Path, default=Path("outputs/experiments_20261017/sentence_chunk_readability_qa.json"))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    if not tokenizer.is_fast:
        raise RuntimeError("The pinned BGE fast tokenizer is required")

    article_count = paragraph_count = long_paragraph_count = chunk_count = oversized_count = 0
    max_observed_tokens = 0
    for path in sorted(args.articles.glob("*.txt")):
        source = path.read_text(encoding="utf-8")
        article_count += 1
        for match in BLOCK_RE.finditer(source):
            paragraph = match.group(0).strip()
            start = match.start()
            if not paragraph:
                continue
            paragraph_count += 1
            if len(tokenizer.encode(paragraph, add_special_tokens=False)) > args.max_unit_tokens:
                long_paragraph_count += 1
            chunks = sentence_preserving_chunks(
                paragraph, start, tokenizer, max_tokens=args.max_unit_tokens
            )
            chunk_count += len(chunks)
            previous_end = start
            for chunk in chunks:
                if source[chunk.start:chunk.end].strip() != chunk.text:
                    raise AssertionError(f"Source mismatch in {path.name} at {chunk.start}")
                if chunk.start != previous_end:
                    raise AssertionError(f"Non-contiguous chunk offsets in {path.name} at {chunk.start}")
                if chunk.token_count > args.max_unit_tokens:
                    if not chunk.oversized_sentence:
                        raise AssertionError("Over-cap chunk lacks the long-sentence flag")
                    oversized_count += 1
                elif chunk.oversized_sentence:
                    raise AssertionError("False long-sentence flag")
                max_observed_tokens = max(max_observed_tokens, chunk.token_count)
                previous_end = chunk.end

    result = {
        "status": "engineering_qa_only_no_gold_labels_read",
        "corpus_used": "previously inspected Evidence Inference article text; only source-text geometry was audited",
        "sentence_segmenter": "PySBD 0.3.4 English char_span=True, clean=False",
        "tokenizer": str(args.tokenizer),
        "unit_cap_tokens": args.max_unit_tokens,
        "articles": article_count,
        "paragraphs": paragraph_count,
        "paragraphs_over_cap": long_paragraph_count,
        "chunks": chunk_count,
        "oversized_single_sentence_chunks": oversized_count,
        "max_chunk_tokens_observed": max_observed_tokens,
        "checks": [
            "chunk text maps back to source offsets",
            "chunk offsets cover each paragraph contiguously",
            "all non-oversized chunks respect the token cap",
            "over-cap units are retained only as intact sentences and flagged",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
