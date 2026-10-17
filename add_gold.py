"""Add trusted essays to the gold set one at a time.

Asks for the prompt, essay, band, optional baseline band and source, checks that
the essay is not in any training data, and appends it to data/gold.csv
(columns: prompt, essay, band, baseline_band, source). eval_gold.py reads the
same file; the gold set is evaluation-only and gitignored.

Training data checked: data/scoring_{train,val}.jsonl, data/tutor_sft.jsonl,
data/my_essays.jsonl, and essays already in data/gold.csv. An essay at or above
--block similarity (TF-IDF cosine, same as prepare_data's --gold-threshold) is
refused; one at or above --warn needs confirmation.

Usage:
  python add_gold.py
"""
import argparse
import csv
import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer

from prepare_data import normalise

COLUMNS = ["prompt", "essay", "band", "baseline_band", "source"]
END = "END"


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_corpus(args):
    """Return [(label, essay)] for every essay that a model may train on."""
    corpus = []
    for path in (args.train, args.val, args.own):
        corpus += [(os.path.basename(path), r["essay"]) for r in read_jsonl(path)]
    for r in read_jsonl(args.tutor):
        user = next(m["content"] for m in r["prompt"] if m["role"] == "user")
        essay = user.split("ESSAY:\n", 1)[-1].split("\n\nSCORER:", 1)[0]
        corpus.append((os.path.basename(args.tutor), essay))
    if os.path.exists(args.gold):
        with open(args.gold, encoding="utf-8", newline="") as f:
            corpus += [("gold.csv (already added)", r["essay"]) for r in csv.DictReader(f)]
    return corpus


def read_block(label):
    print(f"{label} (paste, then a line with only {END}):")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == END:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def read_band(label, optional=False):
    while True:
        raw = input(f"{label}{' (Enter to skip)' if optional else ''}: ").strip()
        if optional and not raw:
            return ""
        try:
            b = float(raw)
        except ValueError:
            print("  enter a number such as 6.5")
            continue
        if 1 <= b <= 9 and b * 2 == int(b * 2):
            return f"{b:.1f}"
        print("  band must be 1-9 in steps of 0.5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/gold.csv")
    ap.add_argument("--train", default="data/scoring_train.jsonl")
    ap.add_argument("--val", default="data/scoring_val.jsonl")
    ap.add_argument("--tutor", default="data/tutor_sft.jsonl")
    ap.add_argument("--own", default="data/my_essays.jsonl")
    ap.add_argument("--block", type=float, default=0.80)
    ap.add_argument("--warn", type=float, default=0.70)
    ap.add_argument("--min-words", type=int, default=120)
    args = ap.parse_args()

    if os.path.exists(args.gold):
        with open(args.gold, encoding="utf-8", newline="") as f:
            header = next(csv.reader(f), [])
        if header != COLUMNS:
            raise SystemExit(f"{args.gold} has columns {header}, expected {COLUMNS}")

    corpus = load_corpus(args)
    print(f"checking against {len(corpus)} essays")
    texts = [normalise(e) for _, e in corpus]
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(texts) if texts else None
    X = vec.transform(texts) if vec else None
    seen = {t: label for t, (label, _) in zip(texts, corpus)}

    added = 0
    while True:
        prompt = read_block("PROMPT")
        essay = read_block("ESSAY")
        if not prompt or not essay:
            print("prompt and essay are required; nothing saved")
        else:
            n_words = len(essay.split())
            if n_words < args.min_words:
                print(f"  ! only {n_words} words (training data uses >= {args.min_words})")
            norm = normalise(essay)
            score, match = 0.0, None
            if norm in seen:
                score, match = 1.0, f"exact duplicate in {seen[norm]}"
            elif vec is not None:
                sims = (vec.transform([norm]) @ X.T).toarray().ravel()
                i = int(sims.argmax())
                score, match = float(sims[i]), corpus[i][0]
            print(f"  most similar: {score:.2f} ({match})")
            ok = True
            if score >= args.block:
                print(f"  refused: similarity >= {args.block}; the essay is already in "
                      "training data or the gold set, so it cannot be an unseen test essay")
                ok = False
            elif score >= args.warn:
                ok = input("  fairly similar to training data; add anyway? [y/N] > ").strip().lower() == "y"

            if ok:
                band = read_band("BAND (examiner / certified teacher)")
                baseline = read_band("BASELINE_BAND from your current tool", optional=True)
                source = ""
                while not source:
                    source = input("SOURCE (who scored it, where it came from): ").strip()
                new_file = not os.path.exists(args.gold)
                os.makedirs(os.path.dirname(args.gold) or ".", exist_ok=True)
                with open(args.gold, "a", encoding="utf-8", newline="") as f:
                    w = csv.writer(f)
                    if new_file:
                        w.writerow(COLUMNS)
                    w.writerow([prompt, essay, band, baseline, source])
                # Later entries in this session are checked against this one too.
                corpus.append(("gold.csv (this session)", essay))
                texts.append(norm)
                seen[norm] = "gold.csv (this session)"
                vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(texts)
                X = vec.transform(texts)
                added += 1
                print(f"  saved to {args.gold}")
        if input("add another? [y/N] > ").strip().lower() != "y":
            break

    print(f"added {added} essays this session")


if __name__ == "__main__":
    main()
