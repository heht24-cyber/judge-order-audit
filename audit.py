"""Exact rank sensitivity to answer order in paired LLM-judge logs."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from itertools import combinations, product
import json
from pathlib import Path


def load(path, anchor):
    """Return common complete blocks, in challenger-first / anchor-first order."""
    digest = hashlib.sha256()
    blocks = defaultdict(dict)
    models, judges = set(), set()
    errors = 0
    with path.open("rb") as source:
        for line_no, line in enumerate(source, 1):
            digest.update(line)
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                first, second = row["model_1"], row["model_2"]
                if first == second or anchor not in (first, second):
                    raise ValueError("each comparison needs the same distinct anchor")
                challenger = second if first == anchor else first
                models.add(challenger)
                judge = row["judge"][0]
                judges.add(judge)
                key = (int(row["question_id"]), int(row["turn"]))
                if challenger in blocks[key]:
                    raise ValueError("duplicate challenger within a question/turn")
                win = "model_2" if first == anchor else "model_1"
                score = {"model_1": 0, "model_2": 0, "tie": 1, "error": None}
                score[win] = 2
                points = tuple(score[row[f"g{i}_winner"]] for i in (1, 2))
                errors += int(None in points)
                blocks[key][challenger] = points[::-1] if first == anchor else points
            except (KeyError, TypeError, ValueError, IndexError) as error:
                raise ValueError(f"line {line_no}: {error}") from error
    if len(judges) != 1 or len(models) < 2:
        raise ValueError("need one judge and at least two challengers")
    complete = sorted(key for key, block in blocks.items()
                      if block.keys() == models and all(None not in p for p in block.values()))
    if not complete:
        raise ValueError("no question/turn has valid verdicts for every challenger")
    model_scores = {m: [blocks[k][m] for k in complete] for m in sorted(models)}
    return complete, model_scores, {
        "sha256": digest.hexdigest(), "anchor": anchor, "judge": judges.pop(),
        "archive_rows": sum(map(len, blocks.values())), "error_rows": errors,
        "archive_blocks": len(blocks), "complete_blocks": len(complete),
        "excluded_blocks": [key for key in sorted(blocks) if key not in complete],
    }


def margin_bounds(a, b):
    # With independent row decisions, A and B have distinct matches.
    lo = sum(min(x) - max(y) for x, y in zip(a, b))
    hi = sum(max(x) - min(y) for x, y in zip(a, b))
    # A block decision selects the same orientation for both challengers.
    block_diffs = [(x[0] - y[0], x[1] - y[1]) for x, y in zip(a, b)]
    return lo, hi, sum(map(min, block_diffs)), sum(map(max, block_diffs))


def reversal(a, b, keys):
    """Smallest set of challenger matches to swap for a strict B-over-A reversal."""
    margin = sum(x[0] - y[0] for x, y in zip(a, b))
    if margin <= 0:
        return None
    losses = []
    for (qid, turn), x, y in zip(keys, a, b):
        if x[0] > x[1]:
            losses.append((x[0] - x[1], qid, turn, "A"))
        if y[0] < y[1]:
            losses.append((y[1] - y[0], qid, turn, "B"))
    losses.sort(key=lambda row: (-row[0], *row[1:]))
    for i, (loss, *_match) in enumerate(losses, 1):
        margin -= loss
        if margin < 0:
            return [row[1:] for row in losses[:i]]
    return None


def rank_rows(scores):
    limits = {m: (sum(map(min, a)), sum(map(max, a))) for m, a in scores.items()}
    denominator = 2 * len(next(iter(scores.values())))
    for model, a in scores.items():
        lo, hi = limits[model]
        best = 1 + sum(other_lo > hi for m, (other_lo, _) in limits.items() if m != model)
        worst = 1 + sum(other_hi > lo for m, (_, other_hi) in limits.items() if m != model)
        first, second = map(sum, zip(*a))
        yield (model, first / denominator, second / denominator,
               (first + second) / (2 * denominator), lo / denominator, hi / denominator,
               best, worst, sum(x != y for x, y in a))


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        writer.writerows(rows)


def audit(source, out):
    keys, scores, summary = source
    ranks = sorted(rank_rows(scores), key=lambda row: (-row[3], row[0]))
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "ranks.csv", ("model", "challenger_first", "anchor_first", "midpoint",
              "score_min", "score_max", "best_rank", "worst_rank", "disagreements"), ranks)
    pair_rows, witnesses = [], []
    counts = Counter()
    denominator = 2 * len(keys)
    for a, b in combinations(scores, 2):
        sa, sb = scores[a], scores[b]
        first = sum(x[0] - y[0] for x, y in zip(sa, sb))
        second = sum(x[1] - y[1] for x, y in zip(sa, sb))
        lo, hi, block_lo, block_hi = margin_bounds(sa, sb)
        row_fixed = lo > 0 or hi < 0
        block_fixed = block_lo > 0 or block_hi < 0
        counts["strict_order_row_robust"] += row_fixed
        counts["strict_order_block_robust"] += block_fixed
        counts["strict_order_flips_between_global_orientations"] += first * second < 0
        counts["both_strict_orders_possible_rowwise"] += lo < 0 < hi
        counts["both_strict_orders_possible_blockwise"] += block_lo < 0 < block_hi
        leader, follower = (a, b) if first >= 0 else (b, a)
        witness = reversal(scores[leader], scores[follower], keys)
        if witness is not None:
            witnesses.append({"leader_first": leader, "follower_first": follower,
                              "swap": [{"question": q, "turn": t,
                                        "model": leader if side == "A" else follower}
                                       for q, t, side in witness]})
        pair_rows.append((a, b, first / denominator, second / denominator,
                          lo / denominator, hi / denominator, block_lo / denominator,
                          block_hi / denominator, len(witness) if witness is not None else ""))
    write_csv(out / "pair_margins.csv", ("model_a", "model_b", "margin_challenger_first",
              "margin_anchor_first", "margin_min", "margin_max", "block_margin_min",
              "block_margin_max", "minimum_swaps_to_reverse_challenger_first"), pair_rows)
    (out / "reversal_witnesses.json").write_text(json.dumps(witnesses, indent=2) + "\n")
    outcomes = Counter(p for a in scores.values() for p in a)
    summary.update(
        challengers=len(scores), retained_rows=sum(outcomes.values()), ordering_pairs=len(pair_rows),
        disagreements=sum(n for (x, y), n in outcomes.items() if x != y),
        opposite_winners=outcomes[(2, 0)] + outcomes[(0, 2)],
        first_position_wins_both_orders=outcomes[(2, 0)],
        second_position_wins_both_orders=outcomes[(0, 2)],
        strict_reversal_witnesses=len(witnesses),
        **counts,
    )
    report = json.dumps(summary, indent=2) + "\n"
    (out / "summary.json").write_text(report, encoding="utf-8")
    print(report, end="")


def self_test():
    a, b = [(2, 0), (2, 1), (0, 2)], [(1, 2), (0, 2), (1, 1)]
    keys = [(101, 1), (101, 2), (102, 1)]
    ra = [sum(x[i] for x, i in zip(a, bits)) for bits in product(range(2), repeat=3)]
    rb = [sum(y[i] for y, i in zip(b, bits)) for bits in product(range(2), repeat=3)]
    coupled = [x - y for x, y in zip(ra, rb)]
    assert margin_bounds(a, b) == (min(ra) - max(rb), max(ra) - min(rb),
                                   min(coupled), max(coupled))
    start = (0, 0, 0, 0, 0, 0)
    feasible_sizes = []
    for bits in product(range(2), repeat=6):
        x = sum(p[i] for p, i in zip(a, bits[:3]))
        y = sum(p[i] for p, i in zip(b, bits[3:]))
        if x < y:
            feasible_sizes.append(sum(i != j for i, j in zip(start, bits)))
    witness = reversal(a, b, keys)
    assert len(witness) == min(feasible_sizes)
    assert reversal([(2, 2)], [(0, 0)], [(0, 1)]) is None
    assert margin_bounds([(2, 0)], [(2, 0)]) == (-2, 2, 0, 0)
    print("Self-test passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, nargs="?")
    parser.add_argument("--anchor", default="gpt-3.5-turbo")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.jsonl is None:
        parser.error("provide the paired judgments JSONL file")
    try:
        audit(load(args.jsonl, args.anchor), args.out)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
