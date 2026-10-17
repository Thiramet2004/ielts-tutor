"""Fine-tune an encoder to predict the overall Writing Task 2 band (regression).

Numbers come from this scorer; words come from the tutor LLM. Keeping the two
apart makes the score measurable and stops the tutor from inventing bands.

Usage:
  python train_scorer.py --model answerdotai/ModernBERT-base
"""
import argparse

import numpy as np
import torch
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

from metrics import band_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/scorer")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()

    ds = load_dataset("json", data_files={
        "train": f"{args.data}/scoring_train.jsonl",
        "validation": f"{args.data}/scoring_val.jsonl",
    })
    tok = AutoTokenizer.from_pretrained(args.model)

    def encode(batch):
        enc = tok(batch["prompt"], batch["essay"], truncation="only_second", max_length=args.max_len)
        enc["labels"] = [float(b) for b in batch["band"]]
        return enc

    ds = ds.map(encode, batched=True, remove_columns=ds["train"].column_names)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, problem_type="regression")

    def compute_metrics(pred):
        return band_metrics(np.asarray(pred.label_ids).reshape(-1),
                            np.asarray(pred.predictions).reshape(-1))

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    targs = TrainingArguments(
        output_dir=args.out,
        learning_rate=args.lr,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="mae",
        greater_is_better=False,
        bf16=bf16,
        logging_steps=50,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    print("validation:", trainer.evaluate())
    trainer.save_model(f"{args.out}/best")
    tok.save_pretrained(f"{args.out}/best")


if __name__ == "__main__":
    main()
