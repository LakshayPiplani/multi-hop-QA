"""debug_infer.py
-------------------------------------------------------------------------
Run a small, interactive/dev‑friendly inference loop that prints **exact**
inputs (prompts), generated raw outputs, and the parsed FINAL answer for a
handful of dev‑split examples.

Usage (from project root):

    python src/debug_infer.py --num 3      # inspect 3 random dev items
    python src/debug_infer.py --start 10 --num 2  # rows 10‑11

-------------------------------------------------------------------------
"""

import os, re, argparse, random, textwrap, torch
from pathlib import Path
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from data_module import load_examples, Example
from graph_builder import build_graph
from token_utilizer_Llama import serialize_example

# ──────────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent.resolve()
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEV_SUBDIR    = "hotpot_dev_distractor_v1"
MODEL_DIR     = PROJECT_ROOT / "models" / "sft2"   # must match train script
MODEL_NAME    = "meta-llama/Llama-3.2-1B"            # same base as training

# helper ───────────────────────────────────────────────────────────────────
ANS_RE = re.compile(r"FINAL:\s*(.+)")

def extract_answer(txt: str) -> str:
    m = ANS_RE.search(txt)
    if not m:
        return "<NONE>"
    candidate = m.group(1)
    # cut at first new line or closing tag
    candidate = re.split(r"\n|\[/INST]|<s>|</s>", candidate, maxsplit=1)[0]
    return candidate.strip()

# ──────────────────────────────────────────────────────────────────────────

def pretty(title: str, content: str, width=88):
    rule = "=" * width
    return f"\n{rule}\n>>> {title}\n{rule}\n{content}\n"


def main(args):
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))

    # ░ Load examples ░
    examples = load_examples(str(PROCESSED_DIR), DEV_SUBDIR)
    if args.shuffle:
        random.shuffle(examples)
    else:
        examples = examples[args.start:]
    examples = examples[: args.num]

    # ░ Model / tokenizer ░
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token  = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map  = "auto"
    )
    model = PeftModel.from_pretrained(base, MODEL_DIR)
    model.eval()

    for idx, ex in enumerate(examples, start=1):
        g = build_graph(ex)
        prompt = serialize_example(ex, g, num_divergent=0).split("CANDIDATE 1:")[0]
        prompt += "[INST]\n"  # ask model to answer

        # Tokenise
        enc = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate (greedy)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False)
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        answer  = extract_answer(decoded)

        # Pretty printing
        print(pretty(f"Example {idx}: Question", ex.question))
        print(pretty("Prompt sent to model", prompt))
        print(pretty("Generated text", decoded))
        print(pretty("Parsed FINAL answer", answer))
        gold = ex.answer or "<NO GOLD>"
        print(pretty("Gold answer", gold))
        print("\n\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Interactive debug inference")
    p.add_argument("--num", type=int, default=3, help="number of samples to show")
    p.add_argument("--start", type=int, default=0, help="start index in dev set")
    p.add_argument("--shuffle", action="store_true", help="random samples")
    p.add_argument("--max_new", type=int, default=64, help="generation tokens")
    args = p.parse_args()
    main(args)
