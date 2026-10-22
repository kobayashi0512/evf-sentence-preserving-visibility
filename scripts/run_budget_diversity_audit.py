#!/usr/bin/env python3
"""Matched-token evidence-selection audit with a BGE+MMR diagnostic.

The script reuses the frozen task definitions and scoring routines from the
original QASPER/SciFact pilots. It adds unsupervised BM25/lexical/lead rankings,
matched token budgets calibrated from the median BM25 top-2/top-4 context
lengths, TF-IDF MMR re-ranking of BGE scores, and a SciFact miss decomposition.
Gold annotations are used only for evaluation, never to select candidates.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from qasper_semantic_visibility_pilot import build_rows as build_qasper_rows
from qasper_semantic_visibility_pilot import evidence_score, score_pairs as score_qasper_pairs, toks
from scifact_semantic_visibility_pilot import load_rows as load_scifact_rows
from scifact_semantic_visibility_pilot import score_pairs as score_scifact_pairs

WORD = re.compile(r"[A-Za-z][A-Za-z0-9+./-]*")
STOP = {
    "the", "and", "for", "with", "from", "that", "this", "were", "was", "are", "is",
    "of", "to", "in", "on", "by", "as", "an", "or", "be", "their", "they", "do", "did",
    "does", "have", "has", "what", "which", "how", "many", "some", "any", "all", "only",
}


def lexical_scores(query: str, candidates: list[str]) -> list[float]:
    q = toks(query)
    return [len(q & toks(text)) / max(1, len(q)) for text in candidates]


def terms(text: str) -> list[str]:
    return [x.lower() for x in WORD.findall(text or "") if len(x) > 2 and x.lower() not in STOP]


def bm25_scores(query: str, candidates: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Document-local BM25 using the archived study's token filter/convention."""
    q = terms(query)
    docs = [terms(text) for text in candidates]
    n = len(docs)
    if not n:
        return []
    avgdl = sum(map(len, docs)) / n or 1.0
    df = Counter(term for doc in docs for term in set(doc))
    scores = []
    for doc in docs:
        tf = Counter(doc)
        score = 0.0
        for term in q:
            f = tf.get(term, 0)
            if not f:
                continue
            idf = math.log1p((n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * len(doc) / avgdl))
        scores.append(score)
    return scores


def stable_order(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda j: (-float(scores[j]), j))


def token_lengths(tokenizer, candidates: list[str]) -> list[int]:
    return [len(tokenizer.encode(text, add_special_tokens=False)) for text in candidates]


def pack_by_budget(order: list[int], lengths: list[int], budget: int) -> list[int]:
    """Keep whole source units; skip a unit that does not fit and continue."""
    selected, used = [], 0
    for idx in order:
        cost = lengths[idx]
        if cost > 0 and used + cost <= budget:
            selected.append(idx)
            used += cost
    return selected


def tfidf_cosine(candidates: list[str]) -> list[list[float]]:
    docs = [Counter(t for t in WORD.findall(text.lower()) if len(t) > 2 and t not in STOP) for text in candidates]
    n = len(docs)
    df = Counter(term for doc in docs for term in doc)
    vectors: list[dict[str, float]] = []
    norms = []
    for doc in docs:
        vec = {term: (1.0 + math.log(freq)) * (math.log((1 + n) / (1 + df[term])) + 1.0)
               for term, freq in doc.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vectors.append(vec)
        norms.append(norm)
    sims = [[0.0] * n for _ in range(n)]
    for i in range(n):
        sims[i][i] = 1.0 if vectors[i] else 0.0
        for j in range(i):
            smaller, larger = (vectors[i], vectors[j]) if len(vectors[i]) < len(vectors[j]) else (vectors[j], vectors[i])
            dot = sum(value * larger.get(term, 0.0) for term, value in smaller.items())
            sim = dot / (norms[i] * norms[j])
            sims[i][j] = sims[j][i] = sim
    return sims


def mmr_order(scores: list[float], candidates: list[str], lengths: list[int], budget: int, lam: float) -> list[int]:
    n = len(scores)
    if n == 0:
        return []
    order = stable_order(scores)
    rel = [0.0] * n
    for rank, idx in enumerate(order):
        rel[idx] = 1.0 if n == 1 else 1.0 - rank / (n - 1)
    sim = tfidf_cosine(candidates)
    selected: list[int] = []
    remaining = set(range(n))
    used = 0
    while remaining:
        feasible = [idx for idx in remaining if lengths[idx] > 0 and used + lengths[idx] <= budget]
        if not feasible:
            break
        def objective(idx: int) -> tuple[float, int]:
            redundancy = max((sim[idx][chosen] for chosen in selected), default=0.0)
            return (lam * rel[idx] - (1.0 - lam) * redundancy, -idx)
        chosen = max(feasible, key=objective)
        selected.append(chosen)
        used += lengths[chosen]
        remaining.remove(chosen)
    return selected


def qasper_row_metrics(selected: list[int], row: dict) -> dict:
    passages = row["paragraphs"]
    selected_text = [passages[j] for j in selected]
    gold = row["evidence"]
    # Exact substring alignment is reported only when every annotated evidence
    # string can be located in at least one source paragraph.
    normalize = lambda s: " ".join((s or "").lower().split())
    gold_to_para = []
    aligned = True
    for item in gold:
        needle = normalize(item)
        hits = [j for j, para in enumerate(passages) if needle and needle in normalize(para)]
        if not hits:
            aligned = False
            break
        gold_to_para.append(set(hits))
    matched = [any(idx in hits for idx in selected) for hits in gold_to_para] if aligned else []
    return {
        "evidence_matching_score": evidence_score(selected_text, gold),
        "gold_text_aligned": aligned,
        "aligned_gold_total": len(gold_to_para) if aligned else None,
        "aligned_gold_hit": sum(matched) if aligned else None,
        "aligned_gold_complete": bool(aligned and all(matched)),
        "aligned_gold_zero_hit": bool(aligned and not any(matched)),
        "aligned_gold_partial": bool(aligned and any(matched) and not all(matched)),
    }


def scifact_row_metrics(selected: list[int], row: dict, lengths: list[int], budget: int) -> dict:
    gold = set(row["gold_sentence_indices"])
    hit = gold & set(selected)
    total_gold_tokens = sum(lengths[idx] for idx in gold)
    feasible = total_gold_tokens <= budget
    n = len(row["sentences"])
    relative_position = (sum(gold) / len(gold)) / max(1, n - 1)
    pos = "early" if relative_position < (1 / 3) else ("middle" if relative_position < (2 / 3) else "late")
    return {
        "recall": len(hit) / len(gold) if gold else 0.0,
        "complete": bool(gold and gold.issubset(set(selected))),
        "zero_hit": not hit,
        "partial": bool(hit and hit != gold),
        "n_gold": len(gold),
        "gold_stratum": "single" if len(gold) == 1 else "multi",
        "position_stratum": pos,
        "relative_position": relative_position,
        "label": row.get("label", "unknown"),
        "budget_feasible": feasible,
        "gold_token_cost": total_gold_tokens,
        "n_selected": len(selected),
        "tokens_used": sum(lengths[idx] for idx in selected),
    }


def bootstrap_diff(rows: list[dict], method: str, baseline: str, metric: str, cluster_key: str, reps: int, seed: int) -> dict:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[str(row[cluster_key])].append(i)
    keys = list(groups)
    if not keys:
        return {"difference": None, "ci95": [None, None], "clusters": 0}
    diffs = np.asarray([row[method][metric] - row[baseline][metric] for row in rows], dtype=float)
    point = float(diffs.mean())
    rng = np.random.default_rng(seed)
    boot = np.empty(reps, dtype=float)
    group_array = np.asarray(keys, dtype=object)
    for b in range(reps):
        sampled = rng.choice(group_array, size=len(group_array), replace=True)
        indices = [i for key in sampled for i in groups[str(key)]]
        boot[b] = float(diffs[indices].mean())
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"difference": point, "ci95": [float(lo), float(hi)], "clusters": len(keys), "bootstrap_reps": reps}


def summarize(records: list[dict], dataset: str, budgets: list[int], reps: int, seed: int) -> dict:
    summary = {"dataset": dataset, "n_instances": len(records), "token_budgets": budgets, "by_budget": {}}
    selectors = sorted({name for record in records for name in record["selectors"]})
    metric_key = "evidence_matching_score" if dataset == "qasper" else "recall"
    cluster_key = "paper_id" if dataset == "qasper" else "claim_id"
    for budget in budgets:
        selected_records = [r for r in records if r["budget"] == budget]
        entry = {"n_instances": len(selected_records), "selectors": {}, "paired_differences": {}}
        for selector in selectors:
            results = [r["selectors"][selector] for r in selected_records]
            keys = results[0].keys() if results else []
            means = {}
            for key in keys:
                vals = [x[key] for x in results if isinstance(x.get(key), (int, float, np.bool_)) and x.get(key) is not None]
                if vals:
                    means[key] = float(np.mean(vals))
            if dataset == "scifact":
                means["n_selected_mean"] = float(np.mean([x["n_selected"] for x in results])) if results else None
                means["tokens_used_mean"] = float(np.mean([x["tokens_used"] for x in results])) if results else None
                feasible = [x for x in results if x["budget_feasible"]]
                means["complete_feasible_only"] = float(np.mean([x["complete"] for x in feasible])) if feasible else None
                means["n_feasible"] = len(feasible)
            else:
                means["aligned_complete_rate"] = float(np.mean([x["aligned_gold_complete"] for x in results if x["gold_text_aligned"]])) if any(x["gold_text_aligned"] for x in results) else None
                means["aligned_zero_hit_rate"] = float(np.mean([x["aligned_gold_zero_hit"] for x in results if x["gold_text_aligned"]])) if any(x["gold_text_aligned"] for x in results) else None
                means["n_exact_aligned"] = sum(x["gold_text_aligned"] for x in results)
            entry["selectors"][selector] = means
        pairs = [("BGE", "BM25"), ("BGE", "MiniLM"), ("BGE_MMR_lam0.7", "BGE"), ("BGE_MMR_lam0.5", "BGE")]
        for method, baseline in pairs:
            if method in selectors and baseline in selectors:
                paired_rows = [{cluster_key: r[cluster_key], method: r["selectors"][method], baseline: r["selectors"][baseline]} for r in selected_records]
                entry["paired_differences"][f"{method}_minus_{baseline}_{metric_key}"] = bootstrap_diff(paired_rows, method, baseline, metric_key, cluster_key, reps, seed + budget)
        if dataset == "scifact":
            entry["failure_strata"] = {}
            for axis in ("gold_stratum", "position_stratum", "label"):
                entry["failure_strata"][axis] = {}
                for level in sorted({r["selectors"]["BGE"][axis] for r in selected_records}):
                    subset = [r["selectors"]["BGE"] for r in selected_records if r["selectors"]["BGE"][axis] == level]
                    feasible = [x for x in subset if x["budget_feasible"]]
                    entry["failure_strata"][axis][str(level)] = {
                        "n": len(subset),
                        "zero_hit_rate": float(np.mean([x["zero_hit"] for x in subset])) if subset else None,
                        "partial_rate": float(np.mean([x["partial"] for x in subset])) if subset else None,
                        "complete_rate": float(np.mean([x["complete"] for x in subset])) if subset else None,
                        "feasible_complete_rate": float(np.mean([x["complete"] for x in feasible])) if feasible else None,
                        "n_budget_feasible": len(feasible),
                    }
        entry["failure_decomposition"] = {}
        if dataset == "scifact":
            bge = [r["selectors"]["BGE"] for r in selected_records]
            entry["failure_decomposition"]["BGE"] = {
                "complete": sum(x["complete"] for x in bge),
                "zero_hit": sum(x["zero_hit"] for x in bge),
                "partial": sum(x["partial"] for x in bge),
                "budget_infeasible": sum(not x["budget_feasible"] for x in bge),
            }
        summary["by_budget"][str(budget)] = entry
    return summary


def evaluate_dataset(name: str, rows: list[dict], model, tokenizer, minilm_model, minilm_tokenizer, device: str, budgets: list[int], max_length: int, batch_size: int, reps: int, seed: int) -> tuple[dict, list[dict]]:
    prepared = []
    # Establish budget anchors from BM25 top-k context lengths using candidate
    # text only. The same integer budgets are then applied to every selector.
    for row in rows:
        candidates = row["paragraphs"] if name == "qasper" else row["sentences"]
        query = row["question"] if name == "qasper" else row["claim"]
        lengths = token_lengths(tokenizer, candidates)
        bm_scores = bm25_scores(query, candidates)
        bm_order = stable_order(bm_scores)
        prepared.append((row, candidates, query, lengths, bm_scores, bm_order))
    calibrated = []
    for k in (2, 4):
        costs = [sum(item[3][idx] for idx in item[5][:k]) for item in prepared]
        calibrated.append(max(1, int(round(statistics.median(costs) / 16) * 16)))
    budgets = sorted(set(budgets or calibrated))
    outputs = []
    torch.set_num_threads(min(4, torch.get_num_threads()))
    for i, (row, candidates, query, lengths, bm_scores, bm_order) in enumerate(prepared, 1):
        if name == "qasper":
            bge_scores = score_qasper_pairs(model, tokenizer, query, candidates, device, batch_size, max_length, "reranker")
            minilm_scores = score_qasper_pairs(minilm_model, minilm_tokenizer, query, candidates, device, batch_size, max_length, "embedding")
        else:
            bge_scores = score_scifact_pairs(model, tokenizer, query, candidates, device, batch_size, max_length, "reranker")
            minilm_scores = score_scifact_pairs(minilm_model, minilm_tokenizer, query, candidates, device, batch_size, max_length, "embedding")
        lex_order = stable_order(lexical_scores(query, candidates))
        bge_order = stable_order(bge_scores)
        minilm_order = stable_order(minilm_scores)
        record_base = {
            "paper_id" if name == "qasper" else "claim_id": row["paper_id"] if name == "qasper" else row["claim_id"],
            "doc_id": row.get("doc_id"),
            "question_id": row.get("question_id"),
            "n_candidates": len(candidates),
            "n_gold": len(row["evidence"]) if name == "qasper" else len(row["gold_sentence_indices"]),
        }
        for budget in budgets:
            orders: dict[str, list[int]] = {
                "Lead": list(range(len(candidates))),
                "Lexical": lex_order,
                "BM25": bm_order,
                "MiniLM": minilm_order,
                "BGE": bge_order,
                "BGE_MMR_lam0.7": mmr_order(bge_scores, candidates, lengths, budget, 0.7),
                "BGE_MMR_lam0.5": mmr_order(bge_scores, candidates, lengths, budget, 0.5),
            }
            selectors = {}
            for selector, order in orders.items():
                selected = order if selector.startswith("BGE_MMR") else pack_by_budget(order, lengths, budget)
                if name == "qasper":
                    metrics = qasper_row_metrics(selected, row)
                    metrics["tokens_used"] = sum(lengths[idx] for idx in selected)
                    metrics["n_selected"] = len(selected)
                else:
                    metrics = scifact_row_metrics(selected, row, lengths, budget)
                metrics["selected_indices"] = selected
                selectors[selector] = metrics
            outputs.append({**record_base, "budget": budget, "selectors": selectors})
        if i % 10 == 0 or i == len(prepared):
            print(f"{name}: scored {i}/{len(prepared)}", flush=True)
    summary = summarize(outputs, name, budgets, reps, seed)
    summary["budget_calibration"] = {
        "method": "median BGE-tokenized passage cost of document-local BM25 top-k; rounded to nearest 16",
        "bm25_top2_median_tokens": calibrated[0],
        "bm25_top4_median_tokens": calibrated[1],
        "budget_tokenizer": "BGE tokenizer, source-unit tokens only; query tokens excluded",
        "selection_policy": "whole source units; ranked candidates that do not fit are skipped; no source text is truncated",
        "mmr_similarity": "within-document TF-IDF cosine, unsupervised",
        "mmr_lambda_values": [0.7, 0.5],
        "paired_bootstrap": "cluster bootstrap, observation-weighted; QASPER by paper_id, SciFact by claim_id",
    }
    return summary, outputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qasper", type=Path, required=True)
    ap.add_argument("--scifact-claims", type=Path, required=True)
    ap.add_argument("--scifact-corpus", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--minilm-model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--qasper-max-length", type=int, default=512)
    ap.add_argument("--scifact-max-length", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--bootstrap-reps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261013)
    args = ap.parse_args()

    qasper_rows = build_qasper_rows(args.qasper, task_only=False)
    qasper_rows.sort(key=lambda r: (r["paper_id"], r["question_id"] or ""))
    scifact_rows = load_scifact_rows(args.scifact_claims, args.scifact_corpus)
    scifact_rows.sort(key=lambda r: (int(r["claim_id"]), int(r["doc_id"]), int(r["rationale_id"])))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(args.model), local_files_only=True).to(device)
    model.eval()
    minilm_tokenizer = AutoTokenizer.from_pretrained(str(args.minilm_model), local_files_only=True)
    minilm_model = AutoModel.from_pretrained(str(args.minilm_model), local_files_only=True).to(device)
    minilm_model.eval()
    args.out.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    all_rows = {}
    for name, rows in (("qasper", qasper_rows), ("scifact", scifact_rows)):
        max_length = args.qasper_max_length if name == "qasper" else args.scifact_max_length
        summary, outputs = evaluate_dataset(name, rows, model, tokenizer, minilm_model, minilm_tokenizer, device, [], max_length, args.batch_size, args.bootstrap_reps, args.seed)
        summary["bge_pair_max_length"] = max_length
        all_summaries[name] = summary
        all_rows[name] = outputs
    (args.out / "matched_token_summary.json").write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out / "matched_token_rows.json").write_text(json.dumps(all_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    brief = {
        name: {
            "n_instances": summary["n_instances"],
            "budgets": summary["token_budgets"],
            "by_budget": {
                budget: {
                    selector: values
                    for selector, values in result["selectors"].items()
                    if selector in ("BM25", "BGE", "BGE_MMR_lam0.7", "BGE_MMR_lam0.5")
                }
                for budget, result in summary["by_budget"].items()
            },
        }
        for name, summary in all_summaries.items()
    }
    print(json.dumps(brief, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
