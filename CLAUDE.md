# CLAUDE.md

Personal IELTS Writing tutor. The owner is a Thai IT student with ML experience,
sitting IELTS in 1-3 months. **Reply to the user in Thai**; keep code, comments and
commit messages in English.

## Architecture
- **Scorer** (`train_scorer.py`): encoder regression (default ModernBERT-base) that
  predicts the overall Writing Task 2 band. All numeric bands come from here.
- **Tutor** (`train_tutor_qlora.py` on CUDA, MLX on Mac): LLM that writes feedback in
  Thai from the essay, the scorer band and a learner profile. It must never invent bands.
- Personalisation comes from the learner profile in the prompt (retrieval), not from
  fine-tuning on the owner's own essays (too few examples).

## Environment
- MacBook Air, Apple Silicon (arm64), 24 GB unified memory. No CUDA.
- Conda env `ielts`, Python 3.11, `pip install -r requirements-mac.txt`.
- Scorer trains on MPS: `PYTORCH_ENABLE_MPS_FALLBACK=1 python train_scorer.py --max-len 1024 --bs 8`.
- Tutor trains with mlx-lm (see README "รันบน Mac"). bitsandbytes/TRL QLoRA only runs on
  an NVIDIA GPU with bf16 (L4/A100, not T4).
- API versions were checked against TRL 1.16, PEFT 0.21, transformers 5.x and
  mlx-lm 0.32. Check the installed version's source before changing trainer arguments.

## Commands
```bash
python prepare_data.py                      # clean + dedupe + group split -> data/scoring_*.jsonl
python train_scorer.py --max-len 1024 --bs 8
python eval_gold.py --gold data/gold.csv    # gate: scorer vs baseline on trusted labels
python generate_tutor_data.py --limit 15    # draft + owner review -> data/tutor_sft.jsonl
python to_mlx_format.py --epochs 2          # data/tutor_sft.jsonl -> data/mlx/{train,valid}.jsonl
```

## Rules
- Never commit anything under `data/` (except the example file), `runs/` or model weights.
  The public dataset has no license; the gold set may contain copyrighted examiner material.
- Never train on `data/gold.csv`. It is evaluation-only.
- Do not use the dataset's `evaluation` column as a training target: it is generic,
  machine-written feedback that often contradicts the band.
- Prefer small, testable changes. Run `python -m py_compile *.py` after editing.
- Before any long training run, do a 1-epoch or small-subset smoke test first.

## Status and next tasks
1. Run `prepare_data.py` and review the duplicate/conflict statistics it prints.
2. Build `data/gold.csv` (50-100 essays with trusted bands; columns prompt, essay, band, baseline_band).
3. Write `generate_tutor_data.py`: draft feedback with a teacher model whose license allows
   training on its outputs, then let the owner review/edit each item before it is saved to
   `data/tutor_sft.jsonl` (TRL conversational prompt-completion format, see the example file).
4. Train the tutor with MLX and compare with the base model on held-out examples.
5. Later: DPO from the owner's chosen/rejected feedback; Speaking pipeline (ASR + fluency features).

Keep the ML work time-boxed (about 1 hour/day) so it does not replace IELTS practice.
