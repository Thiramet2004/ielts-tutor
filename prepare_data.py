"""Clean a public IELTS Writing Task 2 dataset into a scorer training set.

Pipeline
  1. Parse the free-text band labels (e.g. "7.5", "<4") and drop junk rows.
  2. Find exact and near-duplicate essays (TF-IDF cosine). Clusters whose labels
     disagree by more than --max-label-gap are dropped: the same essay cannot be
     both 5.5 and 7.5, so at least one label is wrong.
  3. Split by prompt cluster so the same question never appears in both train
     and validation.

Only the numeric band is kept. The dataset's free-text `evaluation` column is
generic, machine-written feedback that often contradicts the band, so it is not
used as a training target.

Usage:
  python prepare_data.py --dataset chillies/IELTS-writing-task-2-evaluation
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chillies/IELTS-writing-task-2-evaluation")
    ap.add_argument("--prompt-col", default="prompt")
    ap.add_argument("--essay-col", default="essay")
    ap.add_argument("--band-col", default="band")
    ap.add_argument("--out", default="data")
    ap.add_argument("--below4", choices=["drop", "3.5"], default="3.5",
                    help='how to treat labels such as "<4"')
    ap.add_argument("--min-words", type=int, default=120)
    ap.add_argument("--dup-threshold", type=float, default=0.90)
    ap.add_argument("--max-label-gap", type=float, default=1.0)
    ap.add_argument("--prompt-threshold", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = load_dataset(args.dataset)
    rows, raw_count = [], 0
    for split in ds:  # merge all splits, then re-split without leakage
        for ex in ds[split]:
            raw_count += 1
            band = parse_band(ex[args.band_col], args.below4)
            essay = (ex[args.essay_col] or "").strip()
            if band is None or len(essay.split()) < args.min_words:
                continue
            rows.append({"prompt": (ex[args.prompt_col] or "").strip(), "essay": essay, "band": band})
    print(f"rows: {raw_count} raw -> {len(rows)} after label/length filtering")

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
        ngram_range=(1, 1), min_df=1, sublinear_tf=True)
    group_id = np.empty(len(kept), dtype=int)
    for g, idx in enumerate(prompt_groups):
        group_id[idx] = g

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(kept)), groups=group_id))

    os.makedirs(args.out, exist_ok=True)
    for name, idx in (("train", train_idx), ("val", val_idx)):
        path = os.path.join(args.out, f"scoring_{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for i in idx:
                f.write(json.dumps(kept[i], ensure_ascii=False) + "\n")
        print(f"wrote {len(idx):>6} rows -> {path}")

    hist = Counter(r["band"] for r in kept)
    print("band distribution:", dict(sorted(hist.items())))


if __name__ == "__main__":
    main()
