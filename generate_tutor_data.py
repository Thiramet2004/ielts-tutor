"""Draft tutor feedback with a local teacher model, then let the owner review it.

Each item: pick an essay, build the exact prompt the tutor will see at inference
(system prompt + TASK/ESSAY/SCORER/LEARNER PROFILE), let the teacher draft the
completion, run automatic checks, and ask the owner to accept, edit, regenerate
or skip it. Only accepted items reach data/tutor_sft.jsonl (TRL conversational
prompt-completion format, same as data/tutor_sft.example.jsonl).

Teacher: Qwen2.5-14B-Instruct (4-bit MLX conversion). The upstream LICENSE is
plain Apache 2.0 with no extra terms, so its outputs may be used for training.
Check the LICENSE again before switching to another model.

Inputs and outputs
  data/scoring_train.jsonl   essays with cleaned dataset bands (SCORER line)
  data/my_essays.jsonl       optional owner essays: prompt, essay, band, profile
  data/gold.csv              if present, similar essays are excluded (never train on gold)
  data/tutor_sft.jsonl       accepted items (training data)
  data/tutor_drafts.jsonl    every reviewed draft with its decision and metadata;
                             decision "band_wrong" is the log of disputed labels,
                             and draft vs final text of edited items can become DPO pairs

Synthetic learner profiles mix common issues for Thai learners. Some items carry
"decoy" issues that a regex shows are absent from the essay, so the tutor learns
not to link feedback to profile issues the essay does not show.

Usage:
  python generate_tutor_data.py --limit 3 --dry-run   # draft + check, write nothing
  python generate_tutor_data.py --limit 15            # one review session
"""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import tempfile
import time

from prepare_data import normalise

TEACHER = "mlx-community/Qwen2.5-14B-Instruct-4bit"
TEACHER_LICENSE = "apache-2.0"

# Appended to the system prompt for the teacher only; the saved training prompt
# keeps tutor_system_prompt.txt unchanged.
TEACHER_ADDENDUM = """
Additional rules for this draft:
- Only connect feedback to a LEARNER PROFILE issue if you quote an instance of it from this essay. If an issue does not occur, do not claim it does; you may say in one short sentence that it was not seen this time.
- Do not give numeric bands for individual criteria.
- Write every correction as: "exact words from the essay" → "correction", then the explanation in Thai.
- Follow the structure of the example answer."""

# Issues whose presence cannot be checked cheaply. Never used as decoys.
PROFILE_ISSUES = [
    "missing articles before singular countable nouns",
    "subject-verb agreement with uncountable or singular subjects",
    "verb tense consistency within a paragraph",
    "plural -s missing after quantifiers (many, several, a lot of)",
    "run-on sentences joined only with commas",
    "conclusions that introduce new ideas",
    "position not stated clearly in the introduction",
    "body paragraphs without a clear topic sentence",
    "word form errors (e.g. economy/economic, success/successful)",
    "repeating the same noun instead of using pronouns or synonyms",
]

# Issues with a detector: (profile line, regex that finds an instance, mention keywords).
# The keywords are how the check spots feedback that talks about the issue.
DETECTABLE_ISSUES = [
    ("informal contractions (don't, can't, it's)",
     r"\b\w+n't\b|\b(?:i'm|it's|they're|we're|i've|you're|that's)\b", ["contraction"]),
    ("using 'etc.' instead of finishing the list", r"\betc\b", ["etc"]),
    ("starting sentences with And, But or So",
     r"(?:^|[.!?]\s+)(?:and|but|so)\b", ["and, but", "and/but", "but or so", "ขึ้นต้นประโยคด้วย"]),
    ("rhetorical questions in academic writing", r"\?", ["rhetorical", "คำถาม"]),
    ("informal quantifiers such as 'a lot of' and 'lots of'",
     r"\b(?:a lot of|lots of)\b", ["a lot of", "lots of"]),
    ("vague 'very + adjective' instead of precise vocabulary", r"\bvery \w+", ["very"]),
]

CRITERIA = ["Task Response", "Coherence", "Lexical Resource", "Grammatical Range"]
TARGET_BANDS = {7.0}  # the owner's goal may be named as a target
BAND_RE = re.compile(r"(?:band|แบนด์)\s*(\d(?:\.\d)?)", re.I)
QUOTE_RE = re.compile(r'"([^"]+)"|“([^”]+)”')
ARROW_RE = re.compile(r"→|->")
THAI_RE = re.compile(r"[฀-๿]")
LATIN_RE = re.compile(r"[A-Za-z]")


def essay_id(essay):
    return hashlib.sha1(normalise(essay).encode()).hexdigest()[:12]


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def gold_filter(rows, gold_path, threshold=0.9):
    """Drop essays similar to any gold essay (gold is evaluation-only)."""
    if not os.path.exists(gold_path):
        return rows
    from sklearn.feature_extraction.text import TfidfVectorizer
    with open(gold_path, encoding="utf-8") as f:
        gold = [normalise(r["essay"]) for r in csv.DictReader(f)]
    if not gold:
        return rows
    texts = [normalise(r["essay"]) for r in rows]
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(texts + gold)
    sims = (vec.transform(texts) @ vec.transform(gold).T).max(axis=1).toarray().ravel()
    kept = [r for r, s in zip(rows, sims) if s < threshold]
    print(f"gold filter: dropped {len(rows) - len(kept)} essays similar to data/gold.csv")
    return kept


def select_essays(rows, done, rng, focus=(5.0, 7.5), focus_weight=3.0):
    """Band-weighted order without replacement, at most one essay per prompt."""
    seen_prompts, pool = set(), []
    for r in rows:
        p = normalise(r["prompt"])
        if r["id"] in done or p in seen_prompts:
            continue
        seen_prompts.add(p)
        w = focus_weight if focus[0] <= r["band"] <= focus[1] else 1.0
        pool.append((rng.random() ** (1.0 / w), r))  # weighted random order
    pool.sort(key=lambda x: -x[0])
    return [r for _, r in pool]


def make_profile(essay, rng, empty_frac, decoy_frac):
    """Return (profile lines, decoy lines). Decoys are issues absent from the essay."""
    if rng.random() < empty_frac:
        return [], []
    text = essay.lower()
    absent = [(line, kw) for line, rx, kw in DETECTABLE_ISSUES if not re.search(rx, text)]
    present = [line for line, rx, _ in DETECTABLE_ISSUES if re.search(rx, text)]
    decoys = []
    if absent and rng.random() < decoy_frac:
        decoys = rng.sample(absent, k=min(len(absent), rng.choice([1, 1, 2])))
    n_other = rng.randint(1, max(1, 3 - len(decoys)))
    others = rng.sample(PROFILE_ISSUES + present, k=n_other)
    profile = others + [line for line, _ in decoys]
    rng.shuffle(profile)
    return profile, decoys


def user_message(task, essay, band, profile):
    lines = "\n".join(f"- {p}" for p in profile) or "- (none recorded yet)"
    return (f"TASK (Writing Task 2):\n{task}\n\nESSAY:\n{essay}\n\n"
            f"SCORER: overall band {band:.1f} (predicted)\n\n"
            f"LEARNER PROFILE (recurring issues from past essays):\n{lines}")


def _norm_quote(s):
    s = s.replace("’", "'").replace("‘", "'").lower()
    return " ".join(re.sub(r"[^\w' ]+", " ", s).split())


def check_feedback(text, essay, band, profile, decoys):
    """Return a list of warnings in Thai. Warnings never block; the owner decides."""
    warnings = []
    for m in BAND_RE.finditer(text):
        b = float(m.group(1))
        if b != band and b not in TARGET_BANDS:
            warnings.append(f"มีตัวเลข band {m.group(1)} ที่ไม่ตรงกับ SCORER {band:.1f}")

    # Only the quote just left of an arrow claims to be the learner's words;
    # the right side is a correction. A line may hold several arrows.
    essay_n = _norm_quote(essay)
    for line in text.splitlines():
        for left in ARROW_RE.split(line)[:-1]:
            quotes = QUOTE_RE.findall(left)
            if not quotes:
                continue
            quote = quotes[-1][0] or quotes[-1][1]
            pieces = [p for p in re.split(r"\.\.\.|…", quote) if _norm_quote(p)]
            if any(_norm_quote(p) not in essay_n for p in pieces):
                warnings.append(f'คำพูดไม่มีในเรียงความ: "{quote}"')

    if not ARROW_RE.search(text):
        warnings.append('ไม่มีการแก้ในรูปแบบ "คำเดิม" → "คำแก้" จึงตรวจคำพูดไม่ได้')

    lower = text.lower()
    if not profile and "profile" in lower:
        warnings.append("อ้างถึง profile ทั้งที่ profile ว่าง")
    for name in CRITERIA:
        if name.lower() not in lower:
            warnings.append(f"ไม่มีหัวข้อ {name}")
    if not re.search(r"3 จุด|three", lower):
        warnings.append("ไม่มีส่วน 3 จุดที่ควรแก้")
    if "แบบฝึก" not in text:
        warnings.append("ไม่มีแบบฝึกท้าย")
    thai, latin = len(THAI_RE.findall(text)), len(LATIN_RE.findall(text))
    if thai < 0.4 * (thai + latin):
        warnings.append(f"ภาษาไทยน้อย ({thai / max(1, thai + latin):.0%} ของตัวอักษร)")

    for line, keywords in decoys:
        if any(k in lower for k in keywords):
            warnings.append(f"อ้างถึง profile ที่ไม่มีในเรียงความ: {line}")
    return warnings


class Teacher:
    def __init__(self, path, example, temp, max_tokens):
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler
        print(f"loading teacher {path} ...")
        self.model, self.tokenizer = load(path)
        self.example = example
        self.sampler = make_sampler(temp=temp, top_p=0.9)
        self.max_tokens = max_tokens

    def draft(self, system, user):
        from mlx_lm import generate
        ex_user = self.example["prompt"][1]["content"]
        ex_answer = self.example["completion"][0]["content"]
        messages = [{"role": "system", "content": system + TEACHER_ADDENDUM},
                    {"role": "user", "content": ex_user},
                    {"role": "assistant", "content": ex_answer},
                    {"role": "user", "content": user}]
        tokens = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        return generate(self.model, self.tokenizer, tokens, max_tokens=self.max_tokens,
                        sampler=self.sampler).strip()


def edit_in_editor(text):
    with tempfile.NamedTemporaryFile("w+", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(text)
        path = f.name
    try:
        subprocess.call([os.environ.get("EDITOR", "vi"), path])
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    finally:
        os.unlink(path)


def show(item, draft, warnings, profile, decoys):
    print("\n" + "=" * 78)
    print(f"[{item['id']}] source={item['source']} band={item['band']:.1f}")
    print(f"TASK: {item['prompt']}\n\n{item['essay']}\n")
    print("PROFILE:", "; ".join(profile) or "(none)")
    if decoys:
        print("  decoys (not in essay):", "; ".join(line for line, _ in decoys))
    print("-" * 78 + "\n" + draft + "\n" + "-" * 78)
    for w in warnings:
        print("  ! " + w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--essays", default="data/scoring_train.jsonl")
    ap.add_argument("--own", default="data/my_essays.jsonl",
                    help="owner essays (prompt, essay, band, optional profile list); reviewed first")
    ap.add_argument("--gold", default="data/gold.csv")
    ap.add_argument("--system-prompt", default="tutor_system_prompt.txt")
    ap.add_argument("--example", default="data/tutor_sft.example.jsonl")
    ap.add_argument("--out", default="data/tutor_sft.jsonl")
    ap.add_argument("--drafts", default="data/tutor_drafts.jsonl")
    ap.add_argument("--teacher", default=TEACHER)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--limit", type=int, default=15, help="items to process this session")
    ap.add_argument("--empty-profile-frac", type=float, default=0.2)
    ap.add_argument("--decoy-frac", type=float, default=0.5,
                    help="share of non-empty profiles that include an issue absent from the essay")
    ap.add_argument("--dry-run", action="store_true",
                    help="draft and check without review; write nothing")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.system_prompt, encoding="utf-8") as f:
        system = f.read().strip()
    example = read_jsonl(args.example)[0]

    done = {d["id"] for d in read_jsonl(args.drafts)}
    own = [dict(r, id=essay_id(r["essay"]), source="own") for r in read_jsonl(args.own)]
    public = [dict(r, id=essay_id(r["essay"]), source="public") for r in read_jsonl(args.essays)]
    rng = random.Random(args.seed)
    queue = gold_filter([r for r in own if r["id"] not in done], args.gold)
    queue += select_essays(gold_filter(public, args.gold), done, rng)
    print(f"done: {len(done)}; queue: {len(queue)} essays; this session: {args.limit}")

    teacher = Teacher(args.teacher, example, args.temp, args.max_tokens)
    stats = {}
    for item in queue[:args.limit]:
        # Seed per essay so the same essay always gets the same synthetic profile.
        prng = random.Random(f"{args.seed}-{item['id']}")
        if item["source"] == "own" and item.get("profile"):
            profile, decoys = list(item["profile"]), []
        else:
            profile, decoys = make_profile(item["essay"], prng, args.empty_profile_frac, args.decoy_frac)
        user = user_message(item["prompt"], item["essay"], item["band"], profile)

        start = time.time()
        while True:
            t0 = time.time()
            draft = teacher.draft(system, user)
            gen_s = time.time() - t0
            warnings = check_feedback(draft, item["essay"], item["band"], profile, decoys)
            show(item, draft, warnings, profile, decoys)
            if args.dry_run:
                print(f"(dry run: generated in {gen_s:.0f}s, nothing saved)")
                decision, final = "dry_run", None
                break
            choice = input("[a]ccept [e]dit [r]egenerate [s]kip [b]and wrong [q]uit > ").strip().lower()
            if choice == "r":
                continue
            if choice == "e":
                final = edit_in_editor(draft)
                print(final)
                for w in check_feedback(final, item["essay"], item["band"], profile, decoys):
                    print("  ! " + w)
                if input("save edited version? [y/N] > ").strip().lower() != "y":
                    continue
                decision = "accept"
            elif choice == "a":
                decision, final = "accept", draft
            elif choice == "b":
                decision, final = "band_wrong", None
            elif choice == "s":
                decision, final = "skip", None
            elif choice == "q":
                decision = "quit"
            else:
                continue
            break

        if decision == "quit":
            break
        stats[decision] = stats.get(decision, 0) + 1
        if args.dry_run:
            continue
        note = input("note (optional) > ").strip() if decision == "band_wrong" else ""
        append_jsonl(args.drafts, {
            "id": item["id"], "source": item["source"], "band": item["band"],
            "profile": profile, "decoys": [line for line, _ in decoys],
            "teacher": args.teacher, "teacher_license": TEACHER_LICENSE,
            "draft": draft, "final": final, "edited": final is not None and final != draft,
            "warnings": warnings, "decision": decision, "note": note,
            "review_seconds": round(time.time() - start), "time": time.strftime("%Y-%m-%d %H:%M"),
        })
        if decision == "accept":
            append_jsonl(args.out, {
                "prompt": [{"role": "system", "content": system},
                           {"role": "user", "content": user}],
                "completion": [{"role": "assistant", "content": final}],
            })

    print("session:", stats, f"| accepted total: {len(read_jsonl(args.out))}")


if __name__ == "__main__":
    main()
