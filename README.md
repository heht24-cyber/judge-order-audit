# Judge Order Audit

**Which model comparisons survive changes in the order of answers shown to an LLM judge?** This project computes exact score and rank bounds from paired judgments, and emits replayable examples of the smallest set of swaps that reverses a strict ordering.

## Result on MT-Bench

The [public LMSYS archive](https://huggingface.co/spaces/lmsys/mt-bench/blob/6ec03b3cc8ad883fa91c3ca88e0d0ba4f86bb093/data/mt_bench/model_judgment/gpt-4_pair.jsonl) contains 4,800 paired comparisons by a GPT-4 judge. Thirty challengers each faced the same anchor, GPT-3.5-Turbo, on 160 question-turn blocks. Four error rows appeared in three blocks. Requiring a valid pair of verdicts for every challenger left 157 common blocks and 4,710 comparisons. The two recorded orders disagree on 767 comparisons (16.28%); 401 switch between opposite winners.

| Answer-order protocol | Strict pair order survives every allowed assignment | Both strict orders possible |
| --- | ---: | ---: |
| One choice for each challenger match | 261 of 435 (60.0%) | 168 of 435 |
| One shared choice per question and turn | 293 of 435 (67.4%) | 138 of 435 |

A boundary tie accounts for the remaining pairs. Switching globally between challenger-first and anchor-first flips only 14 strict comparisons. Auditing only those two global choices misses much of the sensitivity under either finer-grained protocol.

GPT-4 was first under every assignment. Its score was at least 77.07%, above the next-highest possible score of 75.16% for Claude v1. Claude v1 and Vicuna 33B each ranged from second to fourth. MPT-7B-Chat ranged from eighth to twenty-seventh.

**One-swap witness.** With all challenger answers first, WizardLM-30B leads Guanaco-65B by one half-point over 157 comparisons. Swap only Guanaco's match on question 82, turn 1. Its archived judgment changes from anchor win to Guanaco win, putting Guanaco half a point ahead. WizardLM stays ahead under the all-anchor-first ordering. This is one of 167 strict reversals available from the all-challenger-first baseline. Four need just one swap.

## Reproduce

Download the pinned archive above. SHA-256 of the 48,043,462-byte JSONL:

```text
d662c0b7d1d297f0494fcb4cc09fe8f054fa22d75deb4754a483a921984bc585
```

Python 3.10+ with the standard library:

```sh
python audit.py --self-test
python audit.py gpt-4_pair.jsonl --out results
```

- `summary.json` reports coverage, disagreements and the number of certified pair orders under each protocol.
- `ranks.csv` contains scores for both global orientations, their mean, and attainable rank endpoints under per-match order choices.
- `pair_margins.csv` reports sharp minimum and maximum margins under both protocols. The swap count is blank if the challenger-first baseline is tied or no strict reversal is possible.
- `reversal_witnesses.json` identifies the exact challenger, question and turn to swap. Start from all challenger-first to replay each witness.

## Why the bounds are exact

Use 2 points for a win, 1 for a tie and 0 for a loss, then divide by twice the question-turn count. Each verdict is already mapped to the winning model identity in [FastChat's implementation](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/common.py). The lower and upper per-match score totals sum the smaller and larger verdict scores across blocks, respectively. Comparisons for distinct challengers use disjoint rows, so one challenger's maximum is compatible with every other challenger's minimum, and vice versa. This makes the endpoint competition ranks attainable. The rank is one plus the number of strictly higher scores, so ties share ranks.

For a pair's margin under the shared-block protocol, compute the two score differences within each block and sum their minima or maxima. For a minimum reversal witness under independent choices, sort all margin-decreasing swaps by their decrease and take the shortest prefix that crosses below zero. Every swap costs one match, so a shorter set can never achieve a larger reduction than the same number of largest decreases.

The [MT-Bench paper](https://arxiv.org/abs/2306.05685) documents position bias. This project studies which aggregate ranking claims are invariant to the archived verdicts under explicit order-selection protocols. The intervals are finite-data sensitivity bounds. They provide neither a confidence level nor a measurement of human preference. Separate judging calls can differ for reasons beyond answer position, and the historical GPT-4 judge also evaluates a GPT-4 challenger. Results describe this benchmark, judge and anchor.
