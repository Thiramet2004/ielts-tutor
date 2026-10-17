"""QLoRA SFT for the feedback tutor.

Data: JSONL in TRL's conversational prompt-completion format, one example per line
  {"prompt":     [{"role": "system", "content": ...}, {"role": "user", "content": ...}],
   "completion": [{"role": "assistant", "content": ...}]}
See data/tutor_sft.example.jsonl. For prompt-completion data TRL computes the loss
on the completion only, so the model learns to write feedback, not to copy essays.

Usage (single GPU, roughly 16-24 GB for a 7-8B model at 4096 tokens):
  python train_tutor_qlora.py --model Qwen/Qwen2.5-7B-Instruct --data data/tutor_sft.jsonl
"""
import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def main():
    ap = argparse.ArgumentParser()
    # Any chat model with a Thai-capable tokenizer works; try a newer Qwen or a
    # Typhoon instruct model and keep whichever wins on your held-out examples.
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data", default="data/tutor_sft.jsonl")
    ap.add_argument("--out", default="runs/tutor")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--eval-steps", type=int, default=50)
    args = ap.parse_args()

    ds = load_dataset("json", data_files=args.data)["train"].train_test_split(
        test_size=args.val_frac, seed=42)

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    lora = LoraConfig(
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        max_length=args.max_len,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        model_init_kwargs={"dtype": "bfloat16", "device_map": "auto"},
    )
    trainer = SFTTrainer(
        model=args.model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        peft_config=lora,
        quantization_config=quant,
    )
    trainer.train()
    trainer.save_model(f"{args.out}/adapter")
    print(f"LoRA adapter saved to {args.out}/adapter")


if __name__ == "__main__":
    main()
