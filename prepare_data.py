"""Clean a public IELTS Writing Task 2 dataset into a scorer training set.

Pipeline
  0. Drop raw rows similar to any essay in --gold (TF-IDF cosine >= --gold-threshold),
     so a gold essay can never reach training, whatever the later thresholds are.
  1. Parse the free-text band labels (e.g. "7.5", "<4") and drop junk rows.
  2. Find exact and near-duplicate essays (TF-IDF cosine). Clusters whose labels
     disagree by more than --max-label-gap are dropped: the same essay cannot be
     both 5.5 and 7.5, so at least one label is wrong.
  3. Group prompts by character n-gram TF-IDF similarity and split by group, so
     the same question never appears in both train and validation. Each group's
     side depends only on a hash of its representative prompt (the smallest
     normalised prompt in it): the group goes to validation if the hash falls
     below --val-hash-frac. Removing a few rows (e.g. after adding gold essays)
     therefore leaves every other group on the same side. Groups larger than
     --train-only-frac always go to train so that no question dominates
     validation.

Only the numeric band is kept. The dataset's free-text `evaluation` column is
generic, machine-written feedback that often contradicts the band, so it is not
used as a training target.

Usage:
  python prepare_data.py --dataset chillies/IELTS-writing-task-2-evaluation
"""
import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer

NUM = re.compile(r"\d+(?:\.\d+)?")


def parse_band(raw, below4):
    s = str(raw).strip()
    if s.startswith("<"):
        return None if below4 == "drop" else 3.5
    m = NUM.search(s)
    if not m:
        return None
    b = float(m.group())
    if not 1 <= b <= 9:
        return None
    return round(b * 2) / 2


def normalise(text):
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", str(text).lower()).split())


def similarity_clusters(texts, threshold, chunk=1000, **tfidf_kwargs):
    """Union-find over pairs whose TF-IDF cosine similarity >= threshold."""
    X = TfidfVectorizer(**tfidf_kwargs).fit_transform(texts)  # rows are L2-normalised
    n = X.shape[0]
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for start in range(0, n, chunk):
        sims = (X[start:start + chunk] @ X.T).toarray()
        rows, cols = np.where(sims >= threshold)
        for r, c in zip(rows, cols):
            i, j = start + r, int(c)
            if i < j:
                a, b = find(i), find(j)
                if a != b:
                    parent[b] = a

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def group_hash(key):
    """Stable value in [0, 1) for a string (Python's hash() is salted per process)."""
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) / 2**32


def drop_gold(raw, args):
    """Remove raw rows whose essay is similar to a gold essay (gold is evaluation-only)."""
    if not os.path.exists(args.gold):
        print(f"WARNING: {args.gold} not found; skipping the gold filter. "
              "Re-run this script after building the gold set.")
        return raw
    with open(args.gold, encoding="utf-8", newline="") as f:
        gold = [normalise(r["essay"]) for r in csv.DictReader(f)]
    if not gold:
        print(f"WARNING: {args.gold} is empty; skipping the gold filter.")
        return raw
    texts = [normalise(ex[args.essay_col] or "") for ex in raw]
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(texts + gold)
    sims = (vec.transform(texts) @ vec.transform(gold).T).max(axis=1).toarray().ravel()
    kept = [ex for ex, s in zip(raw, sims) if s < args.gold_threshold]
    print(f"gold filter: dropped {len(raw) - len(kept)} raw rows similar to "
          f"{len(gold)} gold essays (TF-IDF >= {args.gold_threshold})")
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chillies/IELTS-writing-task-2-evaluation")
    ap.add_argument("--prompt-col", default="prompt")
    ap.add_argument("--essay-col", default="essay")
    ap.add_argument("--band-col", default="band")
    ap.add_argument("--out", default="data")
    ap.add_argument("--gold", default="data/gold.csv", help="evaluation-only essays to exclude")
    # Stricter than --dup-threshold: a revised copy of a gold essay (~0.88) must
    # not stay in training, and dropping a few extra training rows costs little.
    ap.add_argument("--gold-threshold", type=float, default=0.80)
    ap.add_argument("--below4", choices=["drop", "3.5"], default="3.5",
                    help='how to treat labels such as "<4"')
    ap.add_argument("--min-words", type=int, default=120)
    ap.add_argument("--dup-threshold", type=float, default=0.90)
    ap.add_argument("--max-label-gap", type=float, default=1.0)
    ap.add_argument("--prompt-threshold", type=float, default=0.8)
    # One question has ~9% of rows even when typo variants are its only merges,
    # so this gate catches transitive chaining, not genuinely popular prompts.
    ap.add_argument("--max-group-frac", type=float, default=0.10,
                    help="abort if the largest prompt group holds this fraction of rows or more")
    ap.add_argument("--train-only-frac", type=float, default=0.02,
                    help="prompt groups larger than this fraction of rows are kept in train")
    # Fraction of eligible groups, not rows. Group sizes are lumpy: 0.08 gives
    # ~9% of rows on chillies/IELTS-writing-task-2-evaluation, 0.09 jumps to ~12.5%.
    ap.add_argument("--val-hash-frac", type=float, default=0.08)
    args = ap.parse_args()

    ds = load_dataset(args.dataset)
    raw = [ex for split in ds for ex in ds[split]]  # merge all splits, re-split later
    raw_count = len(raw)
    raw = drop_gold(raw, args)

    rows = []
    for ex in raw:
        band = parse_band(ex[args.band_col], args.below4)
        essay = (ex[args.essay_col] or "").strip()
        if band is None or len(essay.split()) < args.min_words:
            continue
        rows.append({"prompt": (ex[args.prompt_col] or "").strip(), "essay": essay, "band": band})
    print(f"rows: {raw_count} raw -> {len(raw)} after gold filter -> "
          f"{len(rows)} after label/length filtering")

    clusters = similarity_clusters(
        [normalise(r["essay"]) for r in rows], args.dup_threshold,
        ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    kept, dup_clusters, conflict_rows = [], 0, 0
    for idx in clusters:
        bands = [rows[i]["band"] for i in idx]
        if len(idx) > 1:
            dup_clusters += 1
        if max(bands) - min(bands) > args.max_label_gap:
            conflict_rows += len(idx)
            continue
        r = dict(rows[idx[0]])
        r["band"] = round(float(np.mean(bands)) * 2) / 2
        kept.append(r)
    print(f"duplicate clusters: {dup_clusters}; rows dropped for conflicting labels: {conflict_rows}")
    print(f"rows after de-duplication: {len(kept)}")

    prompt_groups = similarity_clusters(
        [normalise(r["prompt"]) for r in kept], args.prompt_threshold,
        analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    sizes = sorted((len(g) for g in prompt_groups), reverse=True)
    print(f"prompt groups: {len(prompt_groups)}; largest: {sizes[:5]} "
          f"({sizes[0] / len(kept):.1%} of rows); singletons: {sizes.count(1)}")
    if sizes[0] >= args.max_group_frac * len(kept):
        raise SystemExit(f"largest prompt group >= {args.max_group_frac:.0%} of rows; "
                         "raise --prompt-threshold")

    val_idx = []
    for g in prompt_groups:
        if len(g) > args.train_only_frac * len(kept):
            continue
        if group_hash(min(normalise(kept[i]["prompt"]) for i in g)) < args.val_hash_frac:
            val_idx.extend(g)
    val_set = set(val_idx)
    train_idx = [i for i in range(len(kept)) if i not in val_set]
    val_idx = sorted(val_idx)
    for name, idx in (("train", train_idx), ("val", val_idx)):
        n_prompts = len({normalise(kept[i]["prompt"]) for i in idx})
        print(f"{name}: {len(idx)} rows ({len(idx) / len(kept):.1%}), {n_prompts} distinct prompts")

    os.makedirs(args.out, exist_ok=True)
    for name, idx in (("train", train_idx), ("val", val_idx)):
        path = os.path.join(args.out, f"scoring_{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for i in idx:
                f.write(json.dumps(kept[i], ensure_ascii=False) + "\n")
        print(f"wrote {len(idx):>6} rows -> {path}")

    for name, idx in (("all", range(len(kept))), ("train", train_idx), ("val", val_idx)):
        hist = Counter(kept[i]["band"] for i in idx)
        print(f"band distribution ({name}):", dict(sorted(hist.items())))


if __name__ == "__main__":
    main()
