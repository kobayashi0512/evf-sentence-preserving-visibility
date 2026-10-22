#!/usr/bin/env python3
"""Post-hoc paired sensitivity audit restricted to prompts with no BGE pair truncation."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from transformers import AutoTokenizer
from run_evidence_inference_official_dev_confirmation import build_instances
from run_qasper_test_chunk_confirmation import bootstrap_cluster_diff
from src.sentence_chunking import sentence_preserving_chunks

data = ROOT / 'data/evidence_inference_untouched_test'
examples, cohort = build_instances(data)
tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'models/bge-reranker-v2-m3'), local_files_only=True, use_fast=True)
rows = json.loads((ROOT / 'outputs/experiments_20261016/evidence_inference_untouched_test_confirmation/per_prompt.json').read_text())
rows = {r['prompt_id']: r for r in rows if r['budget'] == 240}
eligible = []
eligible_feasible = []
cache = {}
counts = {'paragraph_feasible_pairs': 0, 'paragraph_feasible_truncated': 0, 'chunk_feasible_pairs': 0, 'chunk_feasible_truncated': 0}
for idx, example in enumerate(examples, 1):
    q = example['query']
    paragraphs = example['paragraphs']
    if example['paper_id'] not in cache:
        chunks = []
        for paragraph, (start, _) in zip(paragraphs, example['paragraph_ranges']):
            chunks.extend(piece.text for piece in sentence_preserving_chunks(paragraph, start, tokenizer, max_tokens=240))
        para_costs = [len(x) for x in tokenizer(paragraphs, add_special_tokens=False)['input_ids']]
        chunk_costs = [len(x) for x in tokenizer(chunks, add_special_tokens=False)['input_ids']]
        cache[example['paper_id']] = chunks, para_costs, chunk_costs
    chunks, para_costs, chunk_costs = cache[example['paper_id']]
    para_lengths = [len(x) for x in tokenizer([q] * len(paragraphs), paragraphs, truncation=False)['input_ids']]
    chunk_lengths = [len(x) for x in tokenizer([q] * len(chunks), chunks, truncation=False)['input_ids']]
    if max(para_lengths + chunk_lengths, default=0) <= 512:
        eligible.append(rows[example['prompt_id']])
    for name, costs, lengths in [('paragraph', para_costs, para_lengths), ('chunk', chunk_costs, chunk_lengths)]:
        counts[name + '_feasible_pairs'] += sum(0 < c <= 240 for c in costs)
        counts[name + '_feasible_truncated'] += sum(0 < c <= 240 and n > 512 for c, n in zip(costs, lengths))
    if all(n <= 512 for c, n in zip(para_costs, para_lengths) if 0 < c <= 240) and all(n <= 512 for c, n in zip(chunk_costs, chunk_lengths) if 0 < c <= 240):
        eligible_feasible.append(rows[example['prompt_id']])
    if idx % 100 == 0:
        print(f'checked {idx}/{len(examples)}; eligible {len(eligible)}', flush=True)

out = {'description': 'Exploratory post-hoc subset: all query-candidate pairs in both representations are <=512 BGE tokens, including special tokens. Subset selected without outcome values; selection can alter population composition and does not isolate causation.',
       'budget': 240, 'n_prompts': len(eligible), 'n_articles': len({r['paper_id'] for r in eligible}),
       'total_prompts': len(examples), 'cohort_counts': cohort, 'methods': {}, 'feasible_pair_audit': counts,
       'feasible_no_truncation_prompts': len(eligible_feasible), 'feasible_no_truncation_articles': len({r['paper_id'] for r in eligible_feasible})}
for method in ['BGE', 'BM25']:
    p, c = f'{method}_paragraph', f'{method}_chunk'
    if eligible:
        out['methods'][method] = {'paragraph_mean': sum(r['methods'][p]['character_recall'] for r in eligible)/len(eligible),
                                  'chunk_mean': sum(r['methods'][c]['character_recall'] for r in eligible)/len(eligible),
                                  'paired_bootstrap': bootstrap_cluster_diff([{'paper_id': r['paper_id'], p: r['methods'][p], c: r['methods'][c]} for r in eligible], c, p, 'character_recall', 20000, 20261021) if len({r['paper_id'] for r in eligible}) >= 2 else None}
target = ROOT / 'outputs/experiments_20261016/evf_no_truncation_sensitivity.json'
target.write_text(json.dumps(out, indent=2, ensure_ascii=False)+'\n')
print(json.dumps(out, indent=2), flush=True)
