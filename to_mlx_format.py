"""Convert TRL prompt-completion data to the chat format used by mlx-lm (Apple Silicon).

in : data/tutor_sft.jsonl   {"prompt": [messages...], "completion": [{"role": "assistant", ...}]}
out: data/mlx/train.jsonl and data/mlx/valid.jsonl   {"messages": [prompt..., completion...]}

Train with `mlx_lm.lora ... --mask-prompt` so the loss covers only the final
assistant message, matching the completion-only loss used on CUDA.

Usage:
  python to_mlx_format.py --src data/tutor_sft.jsonl --epochs 2
"""
import argparse
import json
import math
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/tutor_sft.jsonl")
    ap.add_argument("--out", default="data/mlx")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.src, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    chats = [{"messages": r["prompt"] + r["completion"]} for r in rows]
    random.Random(args.seed).shuffle(chats)

    n_val = max(1, int(len(chats) * args.val_frac)) if len(chats) > 1 else 0
    splits = {"valid": chats[:n_val], "train": chats[n_val:]}

    os.makedirs(args.out, exist_ok=True)
    for name, data in splits.items():
        if not data:
            continue
        path = os.path.join(args.out, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(data):>5} rows -> {path}")

    iters = math.ceil(len(splits["train"]) * args.epochs / args.batch_size)
    print(f"mlx-lm counts one batch per iteration: use --iters {iters} "
          f"for {args.epochs:g} epochs at --batch-size {args.batch_size}")


if __name__ == "__main__":
    main()
