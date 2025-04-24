import os, re, json, torch
from pathlib import Path
from tqdm.auto import tqdm
from evaluate import load as load_metric
from transformers import AutoTokenizer, AutoModelForCausalLM

from peft import PeftModel
from data_module import load_examples
from graph_builder import build_graph
from token_utilizer_Llama import serialize_example  # reuse your serializer

# ── Config ───────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent.parent.resolve()
MODEL_DIR      = PROJECT_ROOT / "models" / "sft1"          # must match train_sft.py
BASE_MODEL_ID  = "meta-llama/Llama-3.2-1B"           # download from HF
PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"
DEV_SUBDIR     = "hotpot_dev_distractor_v1"                # evaluate on dev
BATCH_SIZE     = 4                                         # adjust for GPU RAM
MAX_NEW_TOK    = 64                                        # generation budget
SAVE_JSON      = True                                      # write submission.json?

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")

# ── Helper: strip everything after FINAL: ────────────────────────────────────
def extract_answer(text: str) -> str:
    """
    Return the substring after the first 'FINAL:' and before any tag/line break.
    """
    m = re.search(r"FINAL:\s*(.+)", text)
    if not m:
        return ""
    answer = m.group(1).strip()
    answer = re.split(r"(\n|\[/INST]|<s>|</s>)", answer, maxsplit=1)[0]
    return answer.strip().lower()

# ── Load LoRA-tuned model & tokenizer ────────────────────────────────────────
print("Loading fine-tuned model …")
# Load base model from HF and attach LoRA adapter from disk
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="cpu"
)
model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model.eval()

# Load tokenizer from base model ID (to get special tokens, padding, etc.)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

# ── Load dev examples ────────────────────────────────────────────────────────
examples = load_examples(str(PROCESSED_DIR), DEV_SUBDIR)
print(f"Loaded {len(examples)} eval examples")

# ── Metric setup ─────────────────────────────────────────────────────────────
squad_metric = load_metric("squad_v2")

# ── Batched generation loop ─────────────────────────────────────────────────
preds, refs = {}, {}
for i in tqdm(range(0, len(examples), BATCH_SIZE)):
    batch_ex = examples[i : i + BATCH_SIZE]
    prompts = []
    for ex in batch_ex:
        g = build_graph(ex)
        prompt = serialize_example(ex, g, num_divergent=0).split("CANDIDATE 1:")[0]
        prompt += "[INST]\n"
        prompts.append(prompt)

    tok = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **tok,
            max_new_tokens=MAX_NEW_TOK,
            do_sample=False
        )

    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
    for ex, full_text in zip(batch_ex, decoded):
        ans = extract_answer(full_text)
        gold = (ex.answer or "").lower()
        preds[ex.qid] = ans
        refs[ex.qid] = gold
        squad_metric.add(prediction=ans, reference=gold)

# ── Scores ──────────────────────────────────────────────────────────────────
results = squad_metric.compute(
    predictions=list(preds.values()),
    references=list(refs.values())
)
print(f"\nDev Exact Match: {results['exact_match']:.2f}%")
print(f"Dev F1:           {results['f1']:.2f}%")

# ── Optional: write CodaLab submission JSON ─────────────────────────────────
# if SAVE_JSON:
#     submission = {"answer": preds, "sp": {qid: [] for qid in preds}}
#     out_path = PROJECT_ROOT / "submission.json"
#     with open(out_path, "w") as f:
#         json.dump(submission, f, indent=2)
#     print(f"Saved HotpotQA submission to {out_path}")
