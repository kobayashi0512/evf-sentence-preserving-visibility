#!/usr/bin/env python3
"""Exploratory three-way BM25 comparison on EI validation, not held-out test."""
import json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from transformers import AutoTokenizer
from run_evidence_inference_official_dev_confirmation import build_instances
from run_evidence_inference_chunk_granularity import split_long_paragraphs
from run_budget_diversity_audit import bm25_scores, token_lengths, stable_order, pack_by_budget
from run_qasper_test_chunk_confirmation import span_metrics, bootstrap_cluster_diff
from src.sentence_chunking import sentence_preserving_chunks

tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'models/bge-reranker-v2-m3'),local_files_only=True,use_fast=True)
examples,cohort=build_instances(ROOT/'data/evidence_inference_official_dev')
cache={}; rows=[]
for i,ex in enumerate(examples,1):
    pid=ex['paper_id']
    if pid not in cache:
        p=ex['paragraphs']; pr=ex['paragraph_ranges']; sr=[]; st=[]
        for para,(start,end) in zip(p,pr):
            for piece in sentence_preserving_chunks(para,start,tokenizer,max_tokens=240):
                st.append(piece.text);sr.append((piece.start,piece.end))
        # Legacy fixed-width function uses inclusive ends; convert to half-open ranges.
        wt,wr=split_long_paragraphs(p,[(a,b-1) for a,b in pr],tokenizer,240)
        corrected=[]
        for text,(a,b_inclusive) in zip(wt,wr):
            b=b_inclusive+1
            matches=[(paragraph,start) for paragraph,(start,end) in zip(p,pr) if start <= a <= b <= end]
            if len(matches)!=1: raise ValueError('Fixed chunk cannot be mapped to one paragraph')
            paragraph,start=matches[0]; raw=paragraph[a-start:b-start]
            left=len(raw)-len(raw.lstrip()); right=len(raw.rstrip())
            if raw.strip()!=text: raise AssertionError('Chunk text and source offsets disagree')
            corrected.append((a+left,a+right))
        wr=corrected
        cache[pid]={'paragraph':(p,pr,token_lengths(tokenizer,p)), 'sentence':(st,sr,token_lengths(tokenizer,st)), 'fixed':(wt,wr,token_lengths(tokenizer,wt))}
    units=cache[pid]
    unique=list(dict.fromkeys(sum([units[x][0] for x in ('paragraph','sentence','fixed')],[])))
    scores=dict(zip(unique,bm25_scores(ex['query'],unique)))
    row={'paper_id':pid,'prompt_id':ex['prompt_id']}
    for name,(texts,ranges,costs) in units.items():
        sel=pack_by_budget(stable_order([scores[t] for t in texts]),costs,240)
        spans=[ranges[j] for j in sel]
        metrics=[span_metrics(a['spans'],spans) for a in ex['annotations']]
        row[name]={m:float(np.mean([r[m] for r in metrics])) for m in ('character_recall','any_hit','complete')}
        row[name]['tokens_used']=sum(costs[j] for j in sel)
    rows.append(row)
    if i%100==0: print(f'{i}/{len(examples)}',flush=True)
out={'status':'post-hoc exploratory validation, BM25 only; not independent confirmation', 'n_prompts':len(rows),'n_articles':len(cache),'budget':240,'unit_cap':240,'cohort':cohort,'bm25_corpus':'deduplicated union of all three candidate lists per prompt; differs from primary two-list EI corpus', 'means':{}, 'comparisons':{},'rows':rows}
for k in ('paragraph','sentence','fixed'):
    out['means'][k]={m:float(np.mean([r[k][m] for r in rows])) for m in ('character_recall','any_hit','complete','tokens_used')}
for a,b in [('sentence','paragraph'),('fixed','paragraph'),('sentence','fixed')]:
    out['comparisons'][a+'_minus_'+b]=bootstrap_cluster_diff(rows,a,b,'character_recall',20000,20261021)
target=ROOT/'outputs/experiments_20261016/evf_fixedwidth_validation.json'
target.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2),flush=True)
