# Aggregate results reported in the manuscript

These aggregate values match the current manuscript tables. No item-level rows, article text, questions, or rationale annotations are included.

## Held-out evaluations at the 240-token display cap

| Dataset | Selector | Paragraph recall | Chunk recall | Difference (95% CI) | Paragraph any-hit | Chunk any-hit | Paragraph complete | Chunk complete |
|---|---|---:|---:|---|---:|---:|---:|---:|
| QASPER test | BM25 | 0.243 | 0.251 | +0.76 pp (−0.36, +1.88) | 0.289 | 0.301 | 0.214 | 0.218 |
| QASPER test | BGE | 0.360 | 0.391 | +3.15 pp (+1.86, +4.48) | 0.412 | 0.446 | 0.324 | 0.348 |
| Evidence Inference test | BM25 | 0.129 | 0.267 | +13.82 pp (+9.76, +17.96) | 0.134 | 0.299 | 0.119 | 0.219 |
| Evidence Inference test | BGE | 0.187 | 0.324 | +13.70 pp (+9.03, +18.30) | 0.194 | 0.358 | 0.173 | 0.272 |

## Evidence Inference validation budget sensitivity (BGE)

| Budget (tokens) | Paragraph recall (%) | Chunk recall (%) | Difference (95% CI) | Complete-coverage difference | Mean tokens used: paragraph / chunk |
|---:|---:|---:|---|---:|---:|
| 120 | 5.54 | 9.75 | +4.22 pp (+2.75, +5.80) | +1.45 pp | 101 / 110 |
| 180 | 13.56 | 20.72 | +7.16 pp (+4.90, +9.49) | +3.61 pp | 159 / 170 |
| 240 | 19.98 | 36.24 | +16.26 pp (+12.07, +20.41) | +11.07 pp | 216 / 231 |
| 360 | 35.16 | 45.41 | +10.26 pp (+6.11, +14.44) | +4.23 pp | 339 / 350 |
| 480 | 45.09 | 54.82 | +9.73 pp (+5.34, +14.22) | +4.89 pp | 458 / 470 |
| 544 | 47.43 | 58.00 | +10.58 pp (+6.22, +14.99) | +5.80 pp | 518 / 533 |
| 720 | 55.54 | 68.26 | +12.72 pp (+8.40, +17.23) | +8.13 pp | 692 / 707 |
| 960 | 65.24 | 75.40 | +10.16 pp (+6.06, +14.28) | +6.18 pp | 930 / 944 |

The 240-token budget is the prespecified primary operating point, not an empirically selected optimum.

## Length-conditioned trade-off in the Evidence Inference test subset

| Rationale-bearing paragraph class | Prompts | Paragraph recall | Chunk recall | Difference (95% CI) | Paragraph complete | Chunk complete |
|---|---:|---:|---:|---|---:|---:|
| Longer than 240 tokens | 712 | 0.000 | 0.330 | +33.04 pp (+28.69, +37.45) | 0.000 | 0.265 |
| At most 240 tokens | 342 | 0.623 | 0.299 | −32.47 pp (−39.62, −25.47) | 0.599 | 0.282 |
| Mixed long and short | 30 | 0.268 | 0.145 | −12.30 pp (−25.51, +0.28) | 0.000 | 0.000 |

Strata are descriptive and may overlap at the prompt level. These values do not imply that all short-paragraph evidence is lost or that the annotations exhaustively represent scientific evidence.
