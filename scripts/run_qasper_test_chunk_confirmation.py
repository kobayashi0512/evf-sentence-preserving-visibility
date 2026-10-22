#!/usr/bin/env python3
"""Confirm frozen sentence-preserving chunking on the official QASPER test papers."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qasper_semantic_visibility_pilot import score_pairs
from run_budget_diversity_audit import bm25_scores, pack_by_budget, stable_order, token_lengths
from src.sentence_chunking import sentence_preserving_chunks


FLOAT_PREFIX = "FLOAT SELECTED"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_with_map(text: str) -> tuple[str, list[int]]:
    """Lowercase/collapse whitespace while keeping indices into original text."""
    chars: list[str] = []
    offsets: list[int] = []
    in_space = False
    for i, char in enumerate(text):
        if char.isspace():
            if chars and not in_space:
                chars.append(" ")
                offsets.append(i)
            in_space = True
        else:
            lowered = char.lower()
            for out_char in lowered:
                chars.append(out_char)
                offsets.append(i)
            in_space = False
    if chars and chars[-1] == " ":
        chars.pop()
        offsets.pop()
    return "".join(chars), offsets


def occurrences(source: str, needle: str) -> list[tuple[int, int]]:
    source_norm, source_map = normalized_with_map(source)
    needle_norm, _ = normalized_with_map(needle)
    if not needle_norm:
        return []
    found: list[tuple[int, int]] = []
    start = 0
    while True:
        at = source_norm.find(needle_norm, start)
        if at < 0:
            break
        lo = source_map[at]
        hi = source_map[at + len(needle_norm) - 1] + 1
        found.append((lo, hi))
        start = at + max(1, len(needle_norm))
    return found


def flatten_paper(paper: dict) -> tuple[str, list[str], list[tuple[int, int]]]:
    paragraphs: list[str] = []
    for section in paper.get("full_text", []):
        for paragraph in section.get("paragraphs", []):
            if paragraph and paragraph.strip():
                paragraphs.append(paragraph)
    body_parts: list[str] = []
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for paragraph in paragraphs:
        if body_parts:
            body_parts.append("\n\n")
            cursor += 2
        start = cursor
        body_parts.append(paragraph)
        cursor += len(paragraph)
        ranges.append((start, cursor))
    return "".join(body_parts), paragraphs, ranges


def evidence_for_annotation(answer: dict, body: str, paragraphs: list[str], ranges: list[tuple[int, int]]) -> tuple[list[tuple[int, int]] | None, str]:
    evidence = answer.get("evidence") or []
    if isinstance(evidence, str):
        evidence = [evidence]
    textual = [item for item in evidence if isinstance(item, str) and item.strip() and not item.startswith(FLOAT_PREFIX)]
    if not textual:
        return None, "no_text_evidence"

    mapped_paragraphs: set[int] = set()
    for item in textual:
        needle, _ = normalized_with_map(item)
        hits = []
        for idx, paragraph in enumerate(paragraphs):
            haystack, _ = normalized_with_map(paragraph)
            if needle and (needle == haystack or needle in haystack or haystack in needle):
                hits.append(idx)
        if not hits:
            return None, "unaligned_text_evidence"
        mapped_paragraphs.update(hits)

    highlighted = answer.get("highlighted_evidence") or []
    if isinstance(highlighted, str):
        highlighted = [highlighted]
    spans: list[tuple[int, int]] = []
    if highlighted:
        for sentence in highlighted:
            if not isinstance(sentence, str) or not sentence.strip():
                continue
            sentence_hits: list[tuple[int, int]] = []
            for idx in sorted(mapped_paragraphs):
                lo, hi = ranges[idx]
                sentence_hits.extend((lo + a, lo + b) for a, b in occurrences(paragraphs[idx], sentence))
            if not sentence_hits:
                return None, "unaligned_highlighted_evidence"
            spans.extend(sentence_hits)
    else:
        spans.extend(ranges[idx] for idx in sorted(mapped_paragraphs))

    # Deduplicate intervals while retaining the annotated evidence as an
    # alternative annotation, rather than unioning different workers.
    spans = sorted(set(spans))
    return (spans, "aligned") if spans else (None, "empty_text_evidence")


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for lo, hi in sorted(intervals):
        if hi <= lo:
            continue
        if not merged or lo > merged[-1][1]:
            merged.append([lo, hi])
        else:
            merged[-1][1] = max(merged[-1][1], hi)
    return [(lo, hi) for lo, hi in merged]


def span_metrics(gold_spans: list[tuple[int, int]], selected_ranges: list[tuple[int, int]]) -> dict:
    gold = merge_intervals(gold_spans)
    selected = merge_intervals(selected_ranges)
    total = sum(hi - lo for lo, hi in gold)
    overlap = 0
    i = j = 0
    while i < len(gold) and j < len(selected):
        lo, hi = max(gold[i][0], selected[j][0]), min(gold[i][1], selected[j][1])
        if hi > lo:
            overlap += hi - lo
        if gold[i][1] < selected[j][1]:
            i += 1
        else:
            j += 1
    return {
        "character_recall": overlap / total if total else 0.0,
        "any_hit": overlap > 0,
        "complete": overlap == total and total > 0,
        "gold_characters": total,
        "covered_gold_characters": overlap,
    }


def source_text(paper: dict) -> tuple[str, list[str], list[tuple[int, int]]]:
    return flatten_paper(paper)


def build_instances(test_json: dict) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    counts = defaultdict(int)
    for paper_id, paper in test_json.items():
        body, paragraphs, paragraph_ranges = source_text(paper)
        if not paragraphs:
            counts["papers_without_paragraphs"] += 1
            continue
        for qa in paper.get("qas", []):
            counts["questions_total"] += 1
            annotations = []
            for answer_record in qa.get("answers", []):
                answer = answer_record.get("answer", answer_record)
                if answer.get("unanswerable"):
                    counts["unanswerable_annotations"] += 1
                    continue
                spans, status = evidence_for_annotation(answer, body, paragraphs, paragraph_ranges)
                counts[f"annotation_{status}"] += 1
                if spans:
                    annotations.append({
                        "annotation_id": answer_record.get("annotation_id"),
                        "spans": spans,
                    })
            if annotations:
                rows.append({
                    "paper_id": str(paper_id),
                    "question_id": qa.get("question_id"),
                    "question": qa.get("question", ""),
                    "body": body,
                    "paragraphs": paragraphs,
                    "paragraph_ranges": paragraph_ranges,
                    "annotations": annotations,
                })
                counts["questions_in_primary_cohort"] += 1
            else:
                counts["questions_without_aligned_textual_annotation"] += 1
    return rows, dict(counts)


def bootstrap_cluster_diff(rows: list[dict], field_a: str, field_b: str, metric: str,
                           reps: int, seed: int) -> dict:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[row["paper_id"]].append(float(row[field_a][metric]) - float(row[field_b][metric]))
    papers = list(groups)
    if not papers:
        return {"difference": None, "ci95": [None, None], "n_papers": 0}
    point = float(np.mean([value for values in groups.values() for value in values]))
    rng = np.random.default_rng(seed)
    samples = np.empty(reps, dtype=np.float64)
    for b in range(reps):
        selected = rng.choice(papers, size=len(papers), replace=True)
        values = [value for paper in selected for value in groups[str(paper)]]
        samples[b] = np.mean(values) if values else 0.0
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {"difference": point, "ci95": [float(lo), float(hi)], "n_papers": len(papers), "bootstrap_reps": reps}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--bge-model", type=Path, default=Path("models/bge-reranker-v2-m3"))
    parser.add_argument("--out", type=Path, default=Path("outputs/qasper_test_chunk_confirmation"))
    parser.add_argument("--budgets", type=int, nargs="+", default=[240, 544])
    parser.add_argument("--max-unit-tokens", type=int, default=240)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()

    raw = json.loads(args.test_json.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("Expected QASPER v0.3 test JSON as a paper-id keyed object")
    rows, cohort_counts = build_instances(raw)
    if not rows:
        raise RuntimeError(f"No eligible text-evidence questions: {cohort_counts}")

    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(args.bge_model), local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("The frozen offset-preserving protocol requires a fast tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.bge_model), local_files_only=True
    ).to(device).eval()

    scored: list[dict] = []
    truncation = {"paragraph_units": 0, "paragraph_pairs_over_limit": 0,
                  "chunk_units": 0, "chunk_pairs_over_limit": 0}
    long_paragraph_gold: list[int] = []
    for n, row in enumerate(rows, 1):
        chunks = []
        chunk_ranges = []
        oversized = []
        segmentation_fallbacks = []
        for paragraph, (start, _) in zip(row["paragraphs"], row["paragraph_ranges"]):
            pieces = sentence_preserving_chunks(paragraph, start, tokenizer, args.max_unit_tokens)
            for piece in pieces:
                chunks.append(piece.text)
                chunk_ranges.append((piece.start, piece.end))
                oversized.append(piece.oversized_sentence)
                segmentation_fallbacks.append(piece.segmentation_fallback)
        paragraph_lengths = token_lengths(tokenizer, row["paragraphs"])
        chunk_lengths = token_lengths(tokenizer, chunks)
        if any(length > args.max_unit_tokens for i, length in enumerate(chunk_lengths)
               if not oversized[i] and not segmentation_fallbacks[i]):
            raise AssertionError("A non-oversized frozen chunk exceeds the candidate-unit cap")

        for candidate_type, candidates, flag in (
            ("paragraph", row["paragraphs"], "paragraph"),
            ("chunk", chunks, "chunk"),
        ):
            for candidate in candidates:
                encoded = tokenizer(row["question"], candidate, add_special_tokens=True, truncation=False)
                truncation[f"{flag}_units"] += 1
                if len(encoded["input_ids"]) > args.max_length:
                    truncation[f"{flag}_pairs_over_limit"] += 1

        selector_orders = {}
        unit_meta = {}
        for name, candidates, lengths, ranges in (
            ("paragraph", row["paragraphs"], paragraph_lengths, row["paragraph_ranges"]),
            ("chunk", chunks, chunk_lengths, chunk_ranges),
        ):
            bm25_order = stable_order(bm25_scores(row["question"], candidates))
            bge_scores = score_pairs(
                model, tokenizer, row["question"], candidates, device,
                args.batch_size, args.max_length, "reranker",
            )
            selector_orders[name] = {"BM25": bm25_order, "BGE": stable_order(bge_scores)}
            unit_meta[name] = (lengths, ranges)

        for budget in args.budgets:
            method_scores = {}
            method_details = {}
            for unit in ("paragraph", "chunk"):
                lengths, ranges = unit_meta[unit]
                for selector in ("BM25", "BGE"):
                    selected = pack_by_budget(selector_orders[unit][selector], lengths, budget)
                    selected_ranges = [ranges[idx] for idx in selected]
                    annotation_scores = [
                        span_metrics(annotation["spans"], selected_ranges)
                        for annotation in row["annotations"]
                    ]
                    method = f"{selector}_{unit}"
                    method_scores[method] = {
                        metric: float(np.mean([score[metric] for score in annotation_scores]))
                        for metric in ("character_recall", "any_hit", "complete")
                    }
                    method_details[method] = {
                        **method_scores[method],
                        "tokens_used": int(sum(lengths[idx] for idx in selected)),
                        "n_selected": len(selected),
                        "selected_unit_ids": selected,
                        "annotation_metrics": annotation_scores,
                    }
            scored.append({
                "paper_id": row["paper_id"],
                "question_id": row["question_id"],
                "n_annotations": len(row["annotations"]),
                "n_paragraphs": len(row["paragraphs"]),
                "n_chunks": len(chunks),
                "n_oversized_sentences": int(sum(oversized)),
                "n_segmentation_fallbacks": int(sum(segmentation_fallbacks)),
                "budget": budget,
                **method_scores,
                "details": method_details,
            })
        if n % 20 == 0 or n == len(rows):
            print(f"scored eligible question {n}/{len(rows)}", flush=True)

    summary = {
        "status": "frozen_qasper_test_sentence_chunk_confirmation",
        "dataset": "QASPER v0.3 official test split; new source papers in an already studied benchmark family",
        "input_sha256": sha256(args.test_json),
        "n_test_papers": len(raw),
        "n_primary_questions": len(rows),
        "n_primary_papers": len({row["paper_id"] for row in rows}),
        "cohort_counts": cohort_counts,
        "candidate_rule": f"source paragraphs left intact if <= {args.max_unit_tokens} BGE tokens; longer paragraphs split into contiguous complete-sentence groups using PySBD 0.3.4 char spans; over-cap single sentences retained and flagged",
        "selection_rule": "document-local BM25 or zero-shot BGE; whole-unit greedy rank-order packing under identical BGE-token budgets; query and special tokens excluded from displayed-text budget",
        "primary_budget": 240,
        "secondary_budgets": [budget for budget in args.budgets if budget != 240],
        "primary_endpoint": "per-question mean across alternative answer annotations of source-character recall over exact-aligned highlighted textual evidence; fallback to answer evidence paragraphs when highlighted evidence is absent",
        "primary_contrast": "BGE_chunk minus BGE_paragraph character_recall at 240 tokens; paired and paper-clustered",
        "interpretation_boundary": "annotated evidence-text visibility, not evidence sufficiency, answer correctness, reader comprehension, or scientific/clinical decision quality",
        "model": "locally pinned BAAI/bge-reranker-v2-m3; zero-shot, not fine-tuned on QASPER test",
        "bge_pair_truncation_counts": truncation,
        "device": device,
        "budgets": args.budgets,
        "by_budget": {},
        "rows": scored,
    }
    metrics = ("character_recall", "any_hit", "complete")
    for budget in args.budgets:
        subset = [row for row in scored if row["budget"] == budget]
        entry = {"n_questions": len(subset), "methods": {}, "paired_differences": {}}
        for method in ("BM25_paragraph", "BM25_chunk", "BGE_paragraph", "BGE_chunk"):
            entry["methods"][method] = {
                **{metric: float(np.mean([row[method][metric] for row in subset])) for metric in metrics},
                "mean_tokens_used": float(np.mean([row["details"][method]["tokens_used"] for row in subset])),
                "mean_selected_units": float(np.mean([row["details"][method]["n_selected"] for row in subset])),
            }
        for selector in ("BGE", "BM25"):
            for metric in metrics:
                entry["paired_differences"][f"{selector}_chunk_minus_paragraph_{metric}"] = bootstrap_cluster_diff(
                    [{"paper_id": row["paper_id"],
                      "chunk": row[f"{selector}_chunk"],
                      "paragraph": row[f"{selector}_paragraph"]} for row in subset],
                    "chunk", "paragraph", metric, args.bootstrap_reps,
                    args.seed + budget + len(selector) + len(metric),
                )
        summary["by_budget"][str(budget)] = entry

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out / "per_question.json").write_text(json.dumps(scored, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key not in {"rows"}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
