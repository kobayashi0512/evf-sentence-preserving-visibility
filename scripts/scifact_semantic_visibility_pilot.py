#!/usr/bin/env python3
"""SciFact external replication of the evidence-visibility experiment.

The protocol is deliberately document-local: for each claim and each
annotated evidence-containing abstract, rank that abstract's sentences. This
isolates loss/recovery of fine-grained evidence after a document is selected;
it is not open-domain retrieval or claim-veracity evaluation.
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


def load_rows(claims_path: Path, corpus_path: Path) -> list[dict]:
    corpus = {}
    for line in corpus_path.open(encoding="utf-8"):
        d = json.loads(line)
        corpus[int(d["doc_id"])] = d
    rows: list[dict] = []
    for line in claims_path.open(encoding="utf-8"):
        claim = json.loads(line)
        for doc_id, rationales in (claim.get("evidence") or {}).items():
            doc = corpus.get(int(doc_id))
            if not doc or not doc.get("abstract"):
                continue
            # Evaluate each annotated rationale separately; the official data
            # may provide multiple rationale sets for one claim-document pair.
            for ri, rationale in enumerate(rationales):
                gold = sorted(set(int(x) for x in rationale.get("sentences", [])))
                if not gold:
                    continue
                rows.append({
                    "claim_id": claim["id"],
                    "doc_id": int(doc_id),
                    "rationale_id": ri,
                    "label": rationale.get("label"),
                    "claim": claim["claim"],
                    "title": doc.get("title", ""),
                    "sentences": [s.strip() for s in doc["abstract"] if s and s.strip()],
                    "gold_sentence_indices": gold,
                })
    return rows


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-9)


def score_pairs(model, tokenizer, claim: str, sentences: list[str], device: str, batch_size: int, max_length: int, backend: str) -> list[float]:
    scores: list[float] = []
    model.eval()
    with torch.inference_mode():
        if backend == "embedding":
            query = tokenizer(claim, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            query = {k: v.to(device) for k, v in query.items()}
            qvec = torch.nn.functional.normalize(mean_pool(model(**query).last_hidden_state, query["attention_mask"]), p=2, dim=1)
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start : start + batch_size]
            if backend == "reranker":
                encoded = tokenizer([claim] * len(batch), batch, padding=True, truncation=True,
                                     max_length=max_length, return_tensors="pt")
            else:
                encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            if backend == "reranker":
                scores.extend(float(x) for x in model(**encoded).logits.reshape(-1).detach().cpu().tolist())
            else:
                dvec = torch.nn.functional.normalize(mean_pool(model(**encoded).last_hidden_state, encoded["attention_mask"]), p=2, dim=1)
                scores.extend(float(x) for x in (dvec @ qvec.T).reshape(-1).detach().cpu().tolist())
    return scores


def recall_at(selected: list[int], gold: list[int]) -> float:
    return len(set(selected) & set(gold)) / len(set(gold)) if gold else 0.0


def complete_at(selected: list[int], gold: list[int]) -> bool:
    return set(gold).issubset(set(selected)) if gold else False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--backend", choices=["reranker", "embedding"], default="reranker")
    ap.add_argument("--max-claims", type=int, default=0)
    args = ap.parse_args()

    rows = load_rows(args.claims, args.corpus)
    rows.sort(key=lambda r: (int(r["claim_id"]), int(r["doc_id"]), int(r["rationale_id"])))
    if args.max_claims:
        allowed = set(sorted({r["claim_id"] for r in rows})[: args.max_claims])
        rows = [r for r in rows if r["claim_id"] in allowed]

    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    model_cls = AutoModelForSequenceClassification if args.backend == "reranker" else AutoModel
    model = model_cls.from_pretrained(str(args.model), local_files_only=True).to(device)

    out_rows = []
    for i, row in enumerate(rows, 1):
        sentences = row["sentences"]
        claim = row["claim"]
        qtok = toks(claim)
        lexical_order = sorted(range(len(sentences)), key=lambda j: (len(qtok & toks(sentences[j])) / max(1, len(qtok)), -j), reverse=True)
        semantic_scores = score_pairs(model, tokenizer, claim, sentences, device, args.batch_size, args.max_length, args.backend)
        semantic_order = sorted(range(len(sentences)), key=lambda j: (semantic_scores[j], -j), reverse=True)
        gold = row["gold_sentence_indices"]
        # Abstract sentence indices are kept as released; guard malformed
        # indices without silently changing the gold annotation.
        gold = [j for j in gold if 0 <= j < len(sentences)]
        if not gold:
            continue
        out_rows.append({
            "claim_id": row["claim_id"], "doc_id": row["doc_id"], "rationale_id": row["rationale_id"],
            "label": row["label"], "n_sentences": len(sentences), "n_gold_sentences": len(gold),
            "lead1_recall": recall_at(list(range(1)), gold),
            "lead2_recall": recall_at(list(range(min(2, len(sentences)))), gold),
            "lexical2_recall": recall_at(lexical_order[:2], gold),
            "lexical4_recall": recall_at(lexical_order[:4], gold),
            "semantic2_recall": recall_at(semantic_order[:2], gold),
            "semantic4_recall": recall_at(semantic_order[:4], gold),
            "lead2_complete": complete_at(list(range(min(2, len(sentences)))), gold),
            "lexical2_complete": complete_at(lexical_order[:2], gold),
            "lexical4_complete": complete_at(lexical_order[:4], gold),
            "semantic2_complete": complete_at(semantic_order[:2], gold),
            "semantic4_complete": complete_at(semantic_order[:4], gold),
            "gold_sentence_indices": gold,
            "semantic_top4_indices": semantic_order[:4],
        })
        if i % 25 == 0 or i == len(rows):
            print(f"scored {i}/{len(rows)}", flush=True)

    def avg(key: str) -> float:
        return sum(float(r[key]) for r in out_rows) / len(out_rows) if out_rows else 0.0

    summary = {
        "status": "exploratory_scifact_semantic_visibility_pilot",
        "claims": str(args.claims), "corpus": str(args.corpus), "model": str(args.model), "backend": args.backend, "device": device,
        "n_rationales": len(out_rows), "n_claims": len({r["claim_id"] for r in out_rows}),
        "label_counts": dict(Counter(r["label"] for r in out_rows)),
        "means": {k: avg(k) for k in ["lead1_recall", "lead2_recall", "lexical2_recall", "lexical4_recall", "semantic2_recall", "semantic4_recall"]},
        "complete_rates": {k: avg(k) for k in ["lead2_complete", "lexical2_complete", "lexical4_complete", "semantic2_complete", "semantic4_complete"]},
        "interpretation": "Document-local sentence evidence recovery only; no open-domain retrieval, claim-veracity, causal, or decision inference.",
        "rows": out_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["status", "n_rationales", "n_claims", "device", "means", "complete_rates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
