"""Evaluate the scorer on a trusted gold set and compare it with a baseline.

gold.csv columns:
  prompt, essay, band            band = score from an examiner or certified teacher
  baseline_band (optional)       score from your current tool, e.g. a strong
                                 general model prompted with the band descriptors

The gold set must never be used for training. Its labels are the only ones you
fully trust, so this script is the gate that decides which scorer you study with.

Usage:
  python eval_gold.py --scorer runs/scorer/best --gold data/gold.csv
"""
import argparse
import os

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from metrics import band_metrics, snap


def fmt(m):
    return "  ".join(f"{k}={v:.3f}" for k, v in m.items())


@torch.no_grad()
def predict(model, tok, prompts, essays, max_len, bs, device):
    out = []
    for i in range(0, len(essays), bs):
        enc = tok(prompts[i:i + bs], essays[i:i + bs], truncation="only_second",
                  max_length=max_len, padding=True, return_tensors="pt").to(device)
        out.extend(model(**enc).logits.squeeze(-1).float().cpu().tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", default="runs/scorer/best")
    ap.add_argument("--gold", default="data/gold.csv")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", default="runs/gold_predictions.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.gold)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon GPU
    else:
        device = "cpu"
    tok = AutoTokenizer.from_pretrained(args.scorer)
    model = AutoModelForSequenceClassification.from_pretrained(args.scorer).to(device).eval()

    df["scorer_band"] = predict(model, tok, df["prompt"].astype(str).tolist(),
                                df["essay"].astype(str).tolist(), args.max_len, args.bs, device)
    df["scorer_band_snapped"] = snap(df["scorer_band"])
    ours = band_metrics(df["band"], df["scorer_band"])
    print(f"gold set: {len(df)} essays")
    print("scorer   :", fmt(ours))

    # Compare on essays that have a baseline only: a NaN band would make MAE NaN
    # and the verdict below would always favour the baseline.
    has = df["baseline_band"].notna() if "baseline_band" in df.columns else None
    if has is not None and has.any():
        sub = df[has]
        if not has.all():
            print(f"\nbaseline_band present for {has.sum()}/{len(df)} essays; comparing on those")
            ours = band_metrics(sub["band"], sub["scorer_band"])
            print("scorer   :", fmt(ours))
        base = band_metrics(sub["band"], sub["baseline_band"])
        print("baseline :", fmt(base))
        better = ours["mae"] < base["mae"]
        print("verdict  :", "scorer beats baseline - study with it" if better
              else "baseline is still better - keep studying with the baseline")

    df["abs_error"] = (df["scorer_band_snapped"] - df["band"]).abs()
    worst = df.sort_values("abs_error", ascending=False).head(5)
    print("\nlargest errors (inspect these by hand):")
    for _, r in worst.iterrows():
        print(f"  gold={r['band']:.1f}  scorer={r['scorer_band_snapped']:.1f}  essay={r['essay'][:70]!r}...")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\npredictions saved to {args.out}")


if __name__ == "__main__":
    main()
