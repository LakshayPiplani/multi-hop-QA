# ── src/infer_agot.py ─────────────────────────────────────────────────────────
"""
Adaptive GoT inference for GPT-2 LoRA-fine-tuned on HotpotQA.

Prompt format (Llama-style):
<s>[INST] <<SYS>>
You are a careful multi-hop reasoner. Think step-by-step.
<</SYS>>

QUESTION: ...
PARAGRAPHS:
  0: Albert Einstein || Albert Einstein was a German-born ...
  1: Mileva Marić   || Mileva Marić was a Serbian mathematician ...
<|assistant|>
"""

import os, re, math, argparse, torch, json
from collections import deque
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from peft import PeftModel
from evaluate import load as load_metric
from data_module   import load_examples
from graph_builder import build_graph

# ── Hyper-params you can tune at CLI ──────────────────────────────────────────
TAU_BITS   = 1.3   # entropy threshold for branching
MAX_STEPS  = 4     # hard cap on hop length
MAX_NEWTOK = 32    # generation budget per step
MAX_NODES  = 800   # controller queue budget (safety)
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ── Project paths (adjust if needed) ─────────────────────────────────────────
ROOT       = Path(__file__).parent.parent.resolve()
MODEL_DIR  = ROOT / "models" / "sft-gpt2"
PROC_DIR   = ROOT / "data" / "processed"
DEV_SPLIT  = "hotpot_dev_distractor_v1"

# ── Helper: Shannon entropy (bits) of logits row ─────────────────────────────
def entropy_bits(logits):
    p = logits.softmax(-1)
    return -(p * p.log2()).sum().item()

# ── Build initial prompt: question + ALL paragraphs ─────────────────────────
def build_start_prompt(question, graph):
    L = ["<s>[INST] <<SYS>>",
         "You are a careful multi-hop reasoner. Think step-by-step.",
         "<</SYS>>",
         f"\nQUESTION: {question}",
         "PARAGRAPHS:"]
    for pid, data in graph.nodes(data=True):
        sent = " ".join(data["sentences"])
        L.append(f"  {pid}: {data['title']} || {sent}")
    L.append("<|assistant|>")
    return "\n".join(L)

# ── Controller: adaptive DFS/BFS ------------------------------------------------
def agot_answer(model, tok, ex, tau_bits=TAU_BITS):
    graph = build_graph(ex)
    start = build_start_prompt(ex.question, graph)

    Node = tuple[str, list[int], float]          # prompt, chain, logprob
    queue = deque([(start, [], 0.0)])
    completed = []

    while queue and len(completed) < 4 and len(queue) < MAX_NODES:
        prompt, chain, lp = queue.popleft()

        # Stop if exceeded hop limit
        if len(chain) >= MAX_STEPS:
            continue

        inpt = tok(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen = model.generate(
                **inpt,
                max_new_tokens=MAX_NEWTOK,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False
            )

        # Decode ONLY the new text
        new_txt = tok.decode(gen.sequences[0][inpt.input_ids.size(1):],
                             skip_special_tokens=True)

        # --- Check for FINAL answer -----------------------------------------
        m_final = re.search(r"FINAL:\s*(.+)", new_txt)
        if m_final:
            completed.append((chain, m_final.group(1).strip(), lp))
            continue

        # --- Extract predicted next pid -------------------------------------
        m_step = re.search(r"\[STEP\s+\d+\]\s*(\d+)", new_txt)
        if not m_step:
            continue          # malformed → skip
        pid = int(m_step.group(1))

        # Update chain and log-prob of that prediction
        step_logits = gen.scores[0][0]                 # first token logits
        token_id    = tok.encode(str(pid), add_special_tokens=False)[0]
        lp_new      = lp + step_logits.log_softmax(-1)[token_id].item()

        # Compute entropy to decide branching
        H = entropy_bits(step_logits)

        # Append retrieved paragraph text to prompt context
        para = " ".join(graph.nodes[pid]["sentences"])
        new_prompt = (prompt +
                      f"\n[STEP {len(chain)+1}] {pid}"
                      f"\n\nPARA {pid}: {para}\n<|assistant|>")
        queue.append((new_prompt, chain + [pid], lp_new))

        # Branch if LM is uncertain
        if H >= tau_bits and len(queue) < MAX_NODES:
            queue.append((new_prompt, chain + [pid], lp_new - 1.0))

    # Return best answer by log-prob
    if completed:
        return max(completed, key=lambda t: t[2])[1]
    return ""

# ── Main CLI ---------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=200, help="# dev rows to run")
    ap.add_argument("--tau", type=float, default=TAU_BITS, help="entropy gate")
    args = ap.parse_args()

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
    base      = AutoModelForCausalLM.from_pretrained("gpt2", device_map="auto")
    model     = PeftModel.from_pretrained(base, MODEL_DIR).to(DEVICE).eval()

    # Dev data
    examples = load_examples(str(PROC_DIR), DEV_SPLIT)[: args.num]
    metric   = load_metric("squad_v2")

    for ex in tqdm(examples):
        pred = agot_answer(model, tokenizer, ex, tau_bits=args.tau)
        metric.add(prediction=pred.lower(),
                   reference=(ex.answer or "").lower())

    res = metric.compute()
    print(f"Exact Match {res['exact_match']:.2f}%  |  F1 {res['f1']:.2f}%")
