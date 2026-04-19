"""
Verify label consistency in the TriviaPlus dataset.

Chain: sentence-level-word-labels  →  sentence_level_majority_vote  →  response_level_label_binary
      (per-annotator per-sentence)    (majority vote + tiebreak)       (strictest sentence agg)

Step 1: sentence-level-word-labels  →  sentence_level_majority_vote
    For each sentence, take majority vote across annotators.
    Ties broken by strictest label: contradicts > not-mentioned > supports > supplementary.

Step 2: sentence_level_majority_vote  →  response_level_label_binary
    Strictest sentence label determines response label.
    If any sentence is contradicts or not-mentioned → unfaithful (1), else faithful (0).
"""

import sys
from collections import Counter
import pandas as pd

VALID_LABELS = {"supports", "contradicts", "not-mentioned", "supplementary"}

# Tiebreak priority: higher = stricter
TIEBREAK_PRIORITY = {
    "contradicts": 3,
    "not-mentioned": 2,
    "supports": 1,
    "supplementary": 0,
}

# Sentence label → binary response mapping
SENT_TO_BINARY = {
    "contradicts": 1,
    "not-mentioned": 1,
    "supports": 0,
    "supplementary": 0,
}


def majority_vote_with_tiebreak(labels: list[str]) -> str:
    """Majority vote across annotator labels for one sentence.

    On ties, pick the strictest (most unfaithful) label.
    """
    counts = Counter(labels)
    top_count = counts.most_common(1)[0][1]
    tied = [lbl for lbl, cnt in counts.items() if cnt == top_count]
    if len(tied) == 1:
        return tied[0]
    return max(tied, key=lambda x: TIEBREAK_PRIORITY[x])


def aggregate_sentence_to_binary(sentence_labels: list[str]) -> int:
    """Strictest sentence label determines response-level binary label."""
    worst = max(sentence_labels, key=lambda x: TIEBREAK_PRIORITY[x])
    return SENT_TO_BINARY[worst]


def verify(parquet_path: str) -> bool:
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows from {parquet_path}\n")

    # ------------------------------------------------------------------
    # Step 1: sentence-level-word-labels → sentence_level_majority_vote
    # ------------------------------------------------------------------
    step1_total = 0
    step1_match = 0
    step1_mismatches = []

    for idx, row in df.iterrows():
        per_annotator = row["sentence-level-word-labels"]  # array of arrays
        stored = row["sentence_level_majority_vote"]        # list[str]

        for s, stored_label in enumerate(stored):
            # collect valid annotator labels for sentence s
            ann_labels = []
            for a in range(len(per_annotator)):
                if s < len(per_annotator[a]):
                    lbl = per_annotator[a][s]
                    if lbl in VALID_LABELS:
                        ann_labels.append(lbl)
            if not ann_labels:
                continue

            step1_total += 1
            inferred = majority_vote_with_tiebreak(ann_labels)
            if inferred == stored_label:
                step1_match += 1
            else:
                step1_mismatches.append(
                    (idx, s, dict(Counter(ann_labels)), inferred, stored_label)
                )

    print("=" * 60)
    print("Step 1: sentence-level-word-labels → sentence_level_majority_vote")
    print("=" * 60)
    print(f"  Sentences checked : {step1_total}")
    print(f"  Match             : {step1_match} ({step1_match / step1_total * 100:.2f}%)")
    print(f"  Mismatch          : {len(step1_mismatches)}")
    if step1_mismatches:
        for idx, s, votes, inferred, stored in step1_mismatches[:10]:
            print(f"    Row {idx}, Sent {s}: votes={votes} → {inferred}, stored={stored}")
    print()

    # ------------------------------------------------------------------
    # Step 2: sentence_level_majority_vote → response_level_label_binary
    # ------------------------------------------------------------------
    df["inferred_binary"] = df["sentence_level_majority_vote"].apply(
        aggregate_sentence_to_binary
    )
    step2_match = (df["inferred_binary"] == df["response_level_label_binary"]).sum()
    step2_mismatch = len(df) - step2_match

    print("=" * 60)
    print("Step 2: sentence_level_majority_vote → response_level_label_binary")
    print("=" * 60)
    print(f"  Responses checked : {len(df)}")
    print(f"  Match             : {step2_match} ({step2_match / len(df) * 100:.2f}%)")
    print(f"  Mismatch          : {step2_mismatch}")
    if step2_mismatch:
        mis = df[df["inferred_binary"] != df["response_level_label_binary"]]
        for idx, row in mis.head(10).iterrows():
            print(
                f"    Row {idx}: sent_labels={row['sentence_level_majority_vote']}, "
                f"inferred={row['inferred_binary']}, stored={row['response_level_label_binary']}"
            )
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    ok = len(step1_mismatches) == 0 and step2_mismatch == 0
    print("=" * 60)
    if ok:
        print("ALL CHECKS PASSED")
    else:
        print("MISMATCHES FOUND")
    print("=" * 60)
    return ok


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "Triviaplus_all_withnoise_cleaned_20260415.parquet"
    ok = verify(path)
    sys.exit(0 if ok else 1)
