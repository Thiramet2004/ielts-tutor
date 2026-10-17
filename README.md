# IELTS Writing Tutor (personal)

ระบบติวเตอร์ IELTS Writing ส่วนตัว แยกเป็น 2 โมเดลที่ทำงานคนละหน้าที่

```
essay ──► Scorer (encoder regression) ──► overall band ─┐
                                                        ├─► Tutor LLM (QLoRA) ──► ฟีดแบ็กภาษาไทย
learner profile (error log, ดึงจากไฟล์/DB) ─────────────┘
```

**ทำไมแยก:** ตัวเลขมาจาก Scorer ซึ่งวัดผลได้ตรง ๆ ด้วย MAE/QWK ส่วนคำอธิบายมาจาก Tutor ที่ถูกสั่งห้ามแต่ง band เอง วิธีนี้ป้องกันปัญหาที่ LLM ให้คะแนนไม่คงที่ และทำให้ปรับปรุงแต่ละส่วนแยกกันได้

**การปรับให้เป็น "ของคุณ":** ใช้ learner profile ใส่ใน prompt (retrieval) ไม่ใช่ใส่ในน้ำหนักโมเดล เพราะในเวลา 1–3 เดือน คุณจะมีเรียงความของตัวเองแค่หลักสิบชิ้น น้อยเกินไปสำหรับการ fine-tune

## ไฟล์

| ไฟล์ | หน้าที่ |
|---|---|
| `prepare_data.py` | ทำความสะอาด dataset สาธารณะ: แปลง label, ลบ near-duplicate ที่ label ขัดกัน, แบ่ง train/val ตามกลุ่มโจทย์ |
| `train_scorer.py` | เทรน encoder (ค่าเริ่มต้น ModernBERT-base) ให้ทำนาย overall band |
| `eval_gold.py` | วัด Scorer กับ gold set และเทียบกับ baseline (ตัวตัดสินว่าจะใช้อะไรฝึกจริง) |
| `generate_tutor_data.py` | ร่างฟีดแบ็กด้วยโมเดลครู (Qwen2.5-14B-Instruct, Apache 2.0) ผ่าน MLX ตรวจอัตโนมัติ แล้วให้คุณรีวิวก่อนบันทึกลง `data/tutor_sft.jsonl` |
| `train_tutor_qlora.py` | QLoRA SFT ของ Tutor ด้วย TRL รูปแบบ prompt-completion |
| `tutor_system_prompt.txt` | system prompt ของ Tutor ใช้ทั้งตอนสร้างข้อมูลและตอนใช้งาน |
| `data/tutor_sft.example.jsonl` | ตัวอย่าง 1 แถวของข้อมูลเทรน Tutor |
| `metrics.py` | MAE, within-0.5, exact, QWK, Pearson |

## ขั้นตอน

```bash
pip install -r requirements.txt

# 1) เตรียมข้อมูล Scorer
python prepare_data.py --dataset chillies/IELTS-writing-task-2-evaluation

# 2) เทรน Scorer (ราว 8–16 GB VRAM)
python train_scorer.py

# 3) ตัดสินด้วย gold set
python eval_gold.py --gold data/gold.csv

# 4) เทรน Tutor เมื่อมี data/tutor_sft.jsonl แล้ว (ราว 16–24 GB VRAM สำหรับโมเดล 7–8B)
python train_tutor_qlora.py --data data/tutor_sft.jsonl
```

## Gold set: สิ่งที่สำคัญที่สุดในโปรเจกต์นี้

`data/gold.csv` ต้องมีคอลัมน์ `prompt, essay, band` และถ้ามี `baseline_band` (คะแนนจากเครื่องมือที่คุณใช้อยู่) จะได้เทียบกันด้วย

- ใช้เรียงความที่คะแนนเชื่อถือได้จริง 50–100 ชิ้น เช่น ให้ครูที่ผ่านการรับรองตรวจ หรือใช้ตัวอย่างพร้อมความเห็นผู้คุมสอบที่หน่วยงาน IELTS เผยแพร่ (ใช้เป็นชุดทดสอบส่วนตัวเท่านั้น ห้ามนำไปเทรนหรือเผยแพร่ต่อ)
- **ห้ามใช้ gold set ตอนเทรนเด็ดขาด**
- กฎตัดสิน: ถ้า MAE ของ Scorer บน gold set ไม่ต่ำกว่า baseline ให้ฝึกด้วย baseline ต่อไป แล้วค่อยปรับ Scorer ทีหลัง

## ข้อสังเกตเรื่องข้อมูล (ตรวจจากตัวอย่างของ dataset แล้ว)

- **Label noise สูง:** มีเรียงความที่แทบจะเหมือนกันทุกคำแต่ได้ band ต่างกัน 2 ระดับ (เช่น 7.5 กับ 5.5) `prepare_data.py` จึงตัดกลุ่มที่ label ขัดกันทิ้ง ดูตัวเลขที่สคริปต์พิมพ์ออกมาเพื่อประเมินว่า noise มากแค่ไหน
- **คอลัมน์ `evaluation` ไม่ควรใช้เป็นเป้าหมายการเทรน:** เป็นข้อความกว้าง ๆ ที่เครื่องเขียน และหลายแถวชมว่า "effectively addressed" แต่ให้ band 4.5 ถ้าเทรน Tutor ด้วยข้อความนี้ จะได้ฟีดแบ็กแบบเดียวกัน
- **ไม่มี license ระบุ:** ใช้เพื่อการเรียนส่วนตัว อย่าเผยแพร่โมเดลที่เทรนจากข้อมูลนี้ต่อ
- คะแนน validation จาก dataset นี้คือ "ความสอดคล้องกับ label ที่มี noise" ตัวเลขที่เชื่อได้คือผลบน gold set เท่านั้น

## การสร้างข้อมูล Tutor (`data/tutor_sft.jsonl`)

เป้าหมาย 300–800 ตัวอย่างคุณภาพสูงในรูปแบบเดียวกับไฟล์ตัวอย่าง

1. ใช้เรียงความจาก dataset ที่ทำความสะอาดแล้ว บวกเรียงความของคุณเอง
2. ร่างฟีดแบ็กด้วยโมเดลครูที่ license อนุญาตให้นำ output ไปเทรนต่อ (เช่น open model ที่ใช้ Apache 2.0) ถ้าจะใช้ API ของผู้ให้บริการรายใด ให้อ่านข้อตกลงการใช้งานก่อน เพราะหลายเจ้าห้ามนำ output ไปเทรนโมเดลอื่น
3. **แก้ฟีดแบ็กด้วยตัวเอง** ก่อนใส่ในชุดข้อมูล ขั้นนี้คือการเรียน IELTS ไปด้วย เพราะคุณต้องเข้าใจว่าทำไมแต่ละจุดผิด
4. เก็บไว้ 5% เป็นชุดทดสอบ แล้วอ่านผลจริงเทียบกับโมเดลฐานก่อน fine-tune

## ต่อยอด

- **DPO:** ระหว่างฝึกจริง ให้ Tutor สร้างฟีดแบ็ก 2 แบบ แล้วกดเลือกแบบที่ช่วยคุณมากกว่า บันทึกเป็นคู่ chosen/rejected ได้ preference data ของคุณเองฟรี ๆ ใช้กับ `DPOTrainer` ของ TRL
- **Speaking:** ASR ที่ให้ word timestamps → คำนวณ speech rate และสถิติการหยุด (fluency) → ส่ง transcript ให้ Tutor ประเมิน Lexical/Grammar ส่วน pronunciation ยากที่สุด ถ้าจะทำ ให้ดู dataset อย่าง speechocean762 (ผู้พูดภาษาแม่ไม่ใช่อังกฤษ มีคะแนน pronunciation)

## รันบน Mac (Apple Silicon)

| ขั้นตอน | บน Mac | หมายเหตุ |
|---|---|---|
| `prepare_data.py` | ได้ | ใช้ CPU |
| `train_scorer.py` | ได้ | ใช้ GPU ของ Mac ผ่าน MPS ช้ากว่า NVIDIA มาก |
| `eval_gold.py` | ได้ | เลือก MPS อัตโนมัติ |
| `train_tutor_qlora.py` | ไม่ได้ | bitsandbytes ต้องใช้ GPU NVIDIA ให้ใช้ MLX แทน หรือรันสคริปต์นี้บน cloud GPU |

```bash
conda create -n ielts python=3.11 -y
conda activate ielts
pip install -r requirements-mac.txt

python prepare_data.py
PYTORCH_ENABLE_MPS_FALLBACK=1 python train_scorer.py --max-len 512 --bs 8
```

เทรน Tutor ด้วย MLX (หลังจากมี `data/tutor_sft.jsonl` แล้ว):

```bash
mlx_lm.convert --hf-path Qwen/Qwen2.5-7B-Instruct -q --mlx-path models/qwen2.5-7b-4bit
python to_mlx_format.py --epochs 2          # จะพิมพ์ค่า --iters ที่ควรใช้
mlx_lm.lora --model models/qwen2.5-7b-4bit --train --data data/mlx --mask-prompt \
  --batch-size 1 --grad-accumulation-steps 8 --iters <ค่าที่สคริปต์บอก> \
  --learning-rate 1e-5 --num-layers 16 --max-seq-length 3072 --grad-checkpoint \
  --steps-per-eval 50 --save-every 100 --adapter-path runs/tutor-mlx
```

RAM ที่ต้องใช้โดยประมาณ: 8 GB ให้ใช้โมเดลขนาด 1.5–3B, 16 GB ใช้ 7B แบบ 4-bit ได้ถ้าลด `--max-seq-length`, 24 GB ขึ้นไปสบายกว่า

ถ้าจะรัน `train_tutor_qlora.py` บน Google Colab ให้เลือก GPU ที่รองรับ bf16 เช่น L4 หรือ A100 เพราะ T4 ไม่รองรับ bf16
