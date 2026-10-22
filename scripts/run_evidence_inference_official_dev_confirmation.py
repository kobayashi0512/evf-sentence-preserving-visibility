#!/usr/bin/env python3
"""Run the frozen paragraph-versus-sentence-chunk audit on full EI 2.0 dev."""
from __future__ import annotations

import argparse
import csv
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
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from qasper_semantic_visibility_pilot import score_pairs
from run_budget_diversity_audit import bm25_scores, pack_by_budget, stable_order, token_lengths
from scripts.run_qasper_test_chunk_confirmation import bootstrap_cluster_diff, span_metrics
from src.sentence_chunking import sentence_preserving_chunks


BLOCK_RE = re.compile(r"\S.*?(?=\n\s*\n|\Z)", re.S)


def accepted(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def read_source_text(path: Path) -> str:
    # Newline conversion would shift the release's character offsets on CRLF files.
    with path.open(encoding="utf-8", newline="") as handle:
        return handle.read()


def paragraphs_with_offsets(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    matches = list(BLOCK_RE.finditer(text))
    paragraphs, ranges = [], []
    for match in matches:
        value = match.group(0).strip()
        if not value:
            continue
        leading = len(match.group(0)) - len(match.group(0).lstrip())
        start = match.start() + leading
        paragraphs.append(value)
        ranges.append((start, start + len(value)))
    return paragraphs, ranges


def build_instances(data_dir: Path) -> tuple[list[dict], dict]:
    prompt_path = data_dir / "prompts.csv"
    annotation_path = data_dir / "annotations.csv"
    if not prompt_path.exists():
        prompt_path = data_dir / "prompts_validation.csv"
    if not annotation_path.exists():
        annotation_path = data_dir / "annotations_validation.csv"
    with prompt_path.open(encoding="utf-8", newline="") as f:
        prompts = list(csv.DictReader(f))
    with annotation_path.open(encoding="utf-8", newline="") as f:
        annotation_rows = list(csv.DictReader(f))
    grouped: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    text_cache: dict[str, str] = {}
    for row in annotation_rows:
        counts["annotation_rows_total"] += 1
        if not (accepted(row.get("Valid Label")) and accepted(row.get("Valid Reasoning"))):
            counts["annotation_rows_not_jointly_accepted"] += 1
            continue
        counts["annotation_rows_jointly_accepted"] += 1
        try:
            start = int(row.get("Evidence Start", ""))
            end_inclusive = int(row.get("Evidence End", ""))
        except (TypeError, ValueError):
            counts["accepted_rows_missing_offsets"] += 1
            continue
        if start < 0 or end_inclusive < start:
            counts["accepted_rows_invalid_offsets"] += 1
            continue
        pmcid = row.get("PMCID", "").strip()
        if pmcid not in text_cache:
            text_cache[pmcid] = read_source_text(data_dir / "articles" / f"PMC{pmcid}.txt")
        if end_inclusive >= len(text_cache[pmcid]):
            counts["accepted_rows_out_of_bounds"] += 1
            continue
        grouped[(row.get("PromptID", "").strip(), row.get("UserID", "").strip())].append(
            (start, end_inclusive + 1))
        counts["accepted_rows_valid_in_bounds"] += 1

    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for (prompt_id, user_id), spans in grouped.items():
        by_prompt[prompt_id].append({"annotator_id": user_id, "spans": sorted(set(spans))})

    examples = []
    prompt_seen = set()
    for prompt in prompts:
        prompt_id = prompt.get("PromptID", "").strip()
        if prompt_id in prompt_seen:
            counts["duplicate_prompt_rows"] += 1
            continue
        prompt_seen.add(prompt_id)
        annotations = by_prompt.get(prompt_id, [])
        if not annotations:
            counts["prompts_without_accepted_in_bounds_rationale"] += 1
            continue
        pmcid = prompt.get("PMCID", "").strip()
        if pmcid not in text_cache:
            text_cache[pmcid] = read_source_text(data_dir / "articles" / f"PMC{pmcid}.txt")
        text = text_cache[pmcid]
        paragraphs, ranges = paragraphs_with_offsets(text)
        if not paragraphs:
            counts["prompts_without_source_paragraphs"] += 1
            continue
        for annotation in annotations:
            annotation["spans"] = [
                (lo, hi) for lo, hi in annotation["spans"] if hi <= len(text) and lo < hi
            ]
        annotations = [annotation for annotation in annotations if annotation["spans"]]
        if not annotations:
            counts["prompts_without_valid_span_after_alignment"] += 1
            continue
        query = (
            f"With respect to {prompt.get('Outcome', '').strip()}, what difference was reported between "
            f"{prompt.get('Intervention', '').strip()} and {prompt.get('Comparator', '').strip()}?"
        )
        examples.append({
            "paper_id": pmcid,
            "prompt_id": prompt_id,
            "query": query,
            "paragraphs": paragraphs,
            "paragraph_ranges": ranges,
            "annotations": annotations,
        })
        counts["eligible_prompts"] += 1
        counts["eligible_annotator_groups"] += len(annotations)
    return examples, dict(counts)


def rationale_stratum(spans: list[tuple[int, int]], paragraph_ranges: list[tuple[int, int]],
                      paragraph_lengths: list[int], threshold: int = 240) -> str:
    overlapping = set()
    for lo, hi in spans:
        overlapping.update(i for i, (p_lo, p_hi) in enumerate(paragraph_ranges)
                           if max(lo, p_lo) < min(hi, p_hi))
    if not overlapping:
        return "unmapped"
    kinds = {"long" if paragraph_lengths[i] > threshold else "short" for i in overlapping}
    return next(iter(kinds)) if len(kinds) == 1 else "mixed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/evidence_inference_official_dev"))
    parser.add_argument("--bge-model", type=Path, default=Path("models/bge-reranker-v2-m3"))
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/experiments_20261016/evidence_inference_official_dev_confirmation"))
    parser.add_argument("--budgets", type=int, nargs="+", default=[240, 544])
    parser.add_argument("--max-unit-tokens", type=int, default=240)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()

    examples, cohort_counts = build_instances(args.data)
    if not examples:
        raise RuntimeError(f"No eligible source-aligned prompts: {cohort_counts}")
    manifest_path = args.data / "manifest.json"
    dataset_name = (
        json.loads(manifest_path.read_text(encoding="utf-8")).get("dataset")
        if manifest_path.exists() else "Evidence Inference 2.0 prepared split"
    )
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(args.bge_model), local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("Source-offset-preserving segmentation requires a fast tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.bge_model), local_files_only=True
    ).to(device).eval()

    rows = []
    truncation = {"paragraph_pairs": 0, "paragraph_pairs_over_limit": 0,
                  "chunk_pairs": 0, "chunk_pairs_over_limit": 0}
    segmentation = {"oversized_sentences": 0, "fallback_paragraphs": 0,
                    "paragraphs_seen": 0, "chunks_created": 0}
    for n, example in enumerate(examples, 1):
        paragraphs = example["paragraphs"]
        paragraph_ranges = example["paragraph_ranges"]
        chunks, chunk_ranges, overcaps, fallbacks = [], [], [], []
        segmentation["paragraphs_seen"] += len(paragraphs)
        for paragraph, (start, _) in zip(paragraphs, paragraph_ranges):
            for piece in sentence_preserving_chunks(
                    paragraph, start, tokenizer, max_tokens=args.max_unit_tokens):
                chunks.append(piece.text)
                chunk_ranges.append((piece.start, piece.end))
                overcaps.append(piece.oversized_sentence)
                fallbacks.append(piece.segmentation_fallback)
        segmentation["oversized_sentences"] += sum(overcaps)
        segmentation["fallback_paragraphs"] += sum(fallbacks)
        segmentation["chunks_created"] += len(chunks)
        paragraph_lengths = token_lengths(tokenizer, paragraphs)
        chunk_lengths = token_lengths(tokenizer, chunks)
        if any(cost > args.max_unit_tokens and not overcaps[i] and not fallbacks[i]
               for i, cost in enumerate(chunk_lengths)):
            raise AssertionError("A chunk exceeded the frozen cap without a documented exception")

        unique_candidates = list(dict.fromkeys(paragraphs + chunks))
        candidate_scores = score_pairs(
            model, tokenizer, example["query"], unique_candidates, device,
            args.batch_size, args.max_length, "reranker")
        score_by_text = dict(zip(unique_candidates, candidate_scores))
        bm25_by_text = dict(zip(unique_candidates, bm25_scores(example["query"], unique_candidates)))
        methods_base = {}
        unit_data = {
            "paragraph": (paragraphs, paragraph_ranges, paragraph_lengths),
            "chunk": (chunks, chunk_ranges, chunk_lengths),
        }
        for unit_name, (units, ranges, costs) in unit_data.items():
            for unit in units:
                encoded = tokenizer(example["query"], unit, add_special_tokens=True, truncation=False)
                key = "paragraph" if unit_name == "paragraph" else "chunk"
                truncation[f"{key}_pairs"] += 1
                if len(encoded["input_ids"]) > args.max_length:
                    truncation[f"{key}_pairs_over_limit"] += 1
            methods_base[unit_name] = {
                "BGE": stable_order([score_by_text[unit] for unit in units]),
                "BM25": stable_order([bm25_by_text[unit] for unit in units]),
            }

        paragraph_strata = [rationale_stratum(a["spans"], paragraph_ranges, paragraph_lengths)
                             for a in example["annotations"]]
        for budget in args.budgets:
            methods = {}
            strata_scores: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
            for unit_name, (units, ranges, costs) in unit_data.items():
                for selector in ("BM25", "BGE"):
                    selected = pack_by_budget(methods_base[unit_name][selector], costs, budget)
                    selected_ranges = [ranges[i] for i in selected]
                    annotation_scores = [span_metrics(a["spans"], selected_ranges)
                                         for a in example["annotations"]]
                    method = f"{selector}_{unit_name}"
                    methods[method] = {
                        metric: float(np.mean([s[metric] for s in annotation_scores])
                                      if annotation_scores else 0.0)
                        for metric in ("character_recall", "any_hit", "complete")
                    }
                    methods[method].update({
                        "tokens_used": int(sum(costs[i] for i in selected)),
                        "n_selected": len(selected),
                    })
                    prompt_strata: dict[str, list[dict]] = defaultdict(list)
                    for stratum, score in zip(paragraph_strata, annotation_scores):
                        prompt_strata[stratum].append(score)
                    for stratum, scores in prompt_strata.items():
                        strata_scores[method][stratum].append({
                            metric: float(np.mean([score[metric] for score in scores]))
                            for metric in ("character_recall", "any_hit", "complete")
                        })
            row = {
                "paper_id": example["paper_id"],
                "prompt_id": example["prompt_id"],
                "budget": budget,
                "n_annotators": len(example["annotations"]),
                "n_paragraphs": len(paragraphs),
                "n_chunks": len(chunks),
                "methods": methods,
                "strata": {},
            }
            for method, strata in strata_scores.items():
                row["strata"][method] = {
                    stratum: {
                        metric: float(np.mean([s[metric] for s in scores]))
                        for metric in ("character_recall", "any_hit", "complete")
                    }
                    for stratum, scores in strata.items()
                }
            rows.append(row)
        if n % 25 == 0 or n == len(examples):
            print(f"scored {n}/{len(examples)} prompts", flush=True)

    summary = {
        "status": "frozen_zero_shot_out_of_domain_confirmation",
        "dataset": dataset_name,
        "n_eligible_prompts": len(examples),
        "n_source_articles": len({row["paper_id"] for row in examples}),
        "budgets": args.budgets,
        "cohort_counts": cohort_counts,
        "chunk_policy": "unchanged from frozen QASPER confirmation protocol",
        "scoring_max_length": args.max_length,
        "segmentation_counts": segmentation,
        "bge_pair_truncation": truncation,
        "primary_endpoint": "per-prompt mean over accepted annotator groups of source-character recall",
        "primary_contrast": "BGE_chunk minus BGE_paragraph at 240 tokens, paired and clustered by source article",
        "methods": ["BM25_paragraph", "BGE_paragraph", "BM25_chunk", "BGE_chunk"],
        "interpretation_boundary": "annotator-rationale visibility only; not evidence sufficiency, clinical inference accuracy, reader comprehension, or decision quality",
        "device": device,
        "bootstrap_reps": args.bootstrap_reps,
        "seed": args.seed,
        "by_budget": {},
        "rows": rows,
    }
    metrics = ("character_recall", "any_hit", "complete")
    for budget in args.budgets:
        subset = [row for row in rows if row["budget"] == budget]
        entry = {"n_prompts": len(subset), "n_articles": len({r["paper_id"] for r in subset}),
                 "methods": {}, "paired_differences": {}, "rationale_strata": {}}
        method_names = ("BM25_paragraph", "BGE_paragraph", "BM25_chunk", "BGE_chunk")
        for method in method_names:
            entry["methods"][method] = {
                **{metric: float(np.mean([row["methods"][method][metric] for row in subset]))
                   for metric in metrics},
                "mean_tokens_used": float(np.mean([row["methods"][method]["tokens_used"] for row in subset])),
                "mean_units_selected": float(np.mean([row["methods"][method]["n_selected"] for row in subset])),
            }
        for selector in ("BGE", "BM25"):
            for metric in metrics:
                entry["paired_differences"][f"{selector}_chunk_minus_paragraph_{metric}"] = bootstrap_cluster_diff(
                    [{"paper_id": row["paper_id"], "chunk": row["methods"][f"{selector}_chunk"],
                      "paragraph": row["methods"][f"{selector}_paragraph"]} for row in subset],
                    "chunk", "paragraph", metric, args.bootstrap_reps,
                    args.seed + budget + len(selector) + len(metric))
        for stratum in ("long", "short", "mixed", "unmapped"):
            selected_rows = [row for row in subset
                             if row["strata"].get("BGE_chunk", {}).get(stratum) is not None]
            if not selected_rows:
                continue
            entry["rationale_strata"][stratum] = {
                "n_prompts": len(selected_rows),
                "n_articles": len({r["paper_id"] for r in selected_rows}),
                "methods": {}, "paired_differences": {},
            }
            for method in method_names:
                scores = [row["strata"].get(method, {}).get(stratum) for row in selected_rows]
                scores = [score for score in scores if score is not None]
                entry["rationale_strata"][stratum]["methods"][method] = {
                    metric: float(np.mean([score[metric] for score in scores])) for metric in metrics
                }
            for selector in ("BGE", "BM25"):
                paired_rows = [row for row in selected_rows
                               if stratum in row["strata"].get(f"{selector}_chunk", {}) and
                               stratum in row["strata"].get(f"{selector}_paragraph", {})]
                for metric in metrics:
                    entry["rationale_strata"][stratum]["paired_differences"][
                        f"{selector}_chunk_minus_paragraph_{metric}"] = bootstrap_cluster_diff(
                            [{"paper_id": row["paper_id"],
                              "chunk": row["strata"][f"{selector}_chunk"][stratum],
                              "paragraph": row["strata"][f"{selector}_paragraph"][stratum]} for row in paired_rows],
                            "chunk", "paragraph", metric, args.bootstrap_reps,
                            args.seed + budget + len(selector) + len(stratum) + len(metric))
        summary["by_budget"][str(budget)] = entry

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                             encoding="utf-8")
    (args.out / "per_prompt.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
                                                encoding="utf-8")
    print(json.dumps({"status": summary["status"], "by_budget": summary["by_budget"],
                      "cohort_counts": cohort_counts, "segmentation_counts": segmentation,
                      "bge_pair_truncation": truncation}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
