#!/usr/bin/env python3
"""Bounded semantic evidence-retrieval check on the QASPER pilot.

This uses a locally cached BGE reranker only to rank paragraphs for evidence
visibility.  It is not an answer model and its score is not treated as a
truth probability.  The task filter, evidence metric, and held-out dev set
are kept identical to qasper_visibility_pilot.py.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9+./-]*")
TASK = re.compile(
    r"\b(experiment|evaluat|test|validat|ablation|dataset|baseline|compar|benchmark|study)\b",
    re.I,
)
STOP = {
    "the", "and", "for", "with", "from", "that", "this", "were", "was", "are", "is",
    "of", "to", "in", "on", "by", "as", "an", "or", "be", "their", "they", "do", "did",
    "does", "have", "has", "what", "which", "how", "many", "some", "any", "all", "only",
}


def toks(text: str) -> set[str]:
    return {x.lower() for x in TOKEN.findall(text or "") if len(x) > 2 and x.lower() not in STOP}


def f1(a: str, b: str) -> float:
    aa, bb = toks(a), toks(b)
    if not aa or not bb:
        return 0.0
    inter = len(aa & bb)
    if not inter:
        return 0.0
    p, r = inter / len(aa), inter / len(bb)
    return 2 * p * r / (p + r)


def flatten(paper: dict) -> list[str]:
    return [p.strip() for s in paper.get("full_text", []) for p in s.get("paragraphs", []) if p and p.strip()]


def evidence_score(selected: list[str], evidence: list[str]) -> float:
    if not evidence:
        return 0.0
    return sum(max((f1(s, e) for s in selected), default=0.0) for e in evidence) / len(evidence)


def build_rows(path: Path, task_only: bool = True) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for paper_id, paper in data.items():
        paragraphs = flatten(paper)
        for qa in paper.get("qas", []):
            answer = qa.get("answers", [{}])[0].get("answer", {})
            question = qa.get("question", "")
            evidence = answer.get("evidence", []) or []
            if answer.get("unanswerable") or answer.get("yes_no") is None or not evidence or (task_only and not TASK.search(question)):
                continue
            rows.append({
                "paper_id": paper_id,
                "question_id": qa.get("question_id"),
                "question": question,
                "gold_yes_no": bool(answer["yes_no"]),
                "evidence": evidence,
                "paragraphs": paragraphs,
                "abstract": paper.get("abstract", "") or "",
            })
    return rows


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-9)


def score_pairs(model, tokenizer, question: str, paragraphs: list[str], device: str, batch_size: int, max_length: int, backend: str) -> list[float]:
    scores: list[float] = []
    model.eval()
    with torch.inference_mode():
        if backend == "embedding":
            query = tokenizer(question, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            query = {k: v.to(device) for k, v in query.items()}
            qvec = torch.nn.functional.normalize(mean_pool(model(**query).last_hidden_state, query["attention_mask"]), p=2, dim=1)
        for start in range(0, len(paragraphs), batch_size):
            batch = paragraphs[start : start + batch_size]
            if backend == "reranker":
                encoded = tokenizer(
                    [question] * len(batch), batch,
                    padding=True, truncation=True, max_length=max_length,
                    return_tensors="pt",
                )
            else:
                encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            if backend == "reranker":
                logits = model(**encoded).logits.reshape(-1).detach().cpu().tolist()
                scores.extend(float(x) for x in logits)
            else:
                dvec = torch.nn.functional.normalize(mean_pool(model(**encoded).last_hidden_state, encoded["attention_mask"]), p=2, dim=1)
                scores.extend(float(x) for x in (dvec @ qvec.T).reshape(-1).detach().cpu().tolist())
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--backend", choices=["reranker", "embedding"], default="reranker")
    ap.add_argument("--max-questions", type=int, default=0)
    ap.add_argument("--all-yesno", action="store_true", help="Include all answerable yes/no questions as a robustness diagnostic.")
    args = ap.parse_args()

    rows = build_rows(args.input, task_only=not args.all_yesno)
    rows.sort(key=lambda r: (r["paper_id"], r["question_id"] or ""))
    if args.max_questions:
        rows = rows[: args.max_questions]

    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    model_cls = AutoModelForSequenceClassification if args.backend == "reranker" else AutoModel
    model = model_cls.from_pretrained(str(args.model), local_files_only=True)
    model.to(device)

    out_rows = []
    for i, row in enumerate(rows, 1):
        paragraphs = row["paragraphs"]
        scores = score_pairs(model, tokenizer, row["question"], paragraphs, device, args.batch_size, args.max_length, args.backend)
        ranked = sorted(range(len(paragraphs)), key=lambda j: (scores[j], -j), reverse=True)
        # Transparent fixed policies; no gold evidence is used for selection.
        semantic2 = [paragraphs[j] for j in ranked[:2]]
        semantic4 = [paragraphs[j] for j in ranked[:4]]
        # Gold evidence is only used after ranking for evaluation.
        top1_gold = max((f1(paragraphs[ranked[0]], e) for e in row["evidence"]), default=0.0) if ranked else 0.0
        out_rows.append({
            "paper_id": row["paper_id"],
            "question_id": row["question_id"],
            "question": row["question"],
            "gold_yes_no": row["gold_yes_no"],
            "n_paragraphs": len(paragraphs),
            "n_evidence": len(row["evidence"]),
            "abstract_evidence_f1": evidence_score([row["abstract"]], row["evidence"]),
            "lead2_evidence_f1": evidence_score(paragraphs[:2], row["evidence"]),
            "semantic2_evidence_f1": evidence_score(semantic2, row["evidence"]),
            "semantic4_evidence_f1": evidence_score(semantic4, row["evidence"]),
            "semantic_top1_max_evidence_f1": top1_gold,
            "semantic_top1_index": int(ranked[0]) if ranked else None,
            "semantic_top2_indices": [int(x) for x in ranked[:2]],
            "lexical2_evidence_f1": evidence_score(
                sorted(paragraphs, key=lambda p: (len(toks(row["question"]) & toks(p)) / max(1, len(toks(row["question"]))), -paragraphs.index(p)), reverse=True)[:2], row["evidence"]
            ),
            "lexical4_evidence_f1": evidence_score(
                sorted(paragraphs, key=lambda p: (len(toks(row["question"]) & toks(p)) / max(1, len(toks(row["question"]))), -paragraphs.index(p)), reverse=True)[:4], row["evidence"]
            ),
        })
        print(f"scored {i}/{len(rows)}", flush=True)

    def avg(key: str) -> float:
        return sum(float(r[key]) for r in out_rows) / len(out_rows) if out_rows else 0.0

    summary = {
        "status": "exploratory_semantic_visibility_pilot",
        "input": str(args.input),
        "model": str(args.model),
        "backend": args.backend,
        "task": "all answerable yes/no questions (robustness diagnostic)" if args.all_yesno else "yes/no questions mentioning experiments, evaluation, baselines, datasets, or comparisons",
        "device": device,
        "n_questions": len(out_rows),
        "n_papers": len({r["paper_id"] for r in out_rows}),
        "gold_answer_counts": dict(Counter(str(r["gold_yes_no"]).lower() for r in out_rows)),
        "means": {k: avg(k) for k in ["abstract_evidence_f1", "lead2_evidence_f1", "lexical2_evidence_f1", "lexical4_evidence_f1", "semantic2_evidence_f1", "semantic4_evidence_f1", "semantic_top1_max_evidence_f1"]},
        "interpretation": "BGE is evaluated only as a paragraph relevance ranker. Results are evidence-selection diagnostics on a small held-out pilot, not answer accuracy, causal evidence, or decision-support evidence.",
        "rows": out_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["status", "n_questions", "n_papers", "device", "means"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
