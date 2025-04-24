import os
from pathlib import Path
from typing import Tuple, List
import networkx as nx
from transformers import LlamaTokenizer
from data_module import load_examples, Example
import random

# Compute project root based on this file's location
def get_project_root() -> Path:
    return Path(__file__).parent.parent.resolve()

# Default processed data subdirectories
def get_processed_dirs() -> list[str]:
    return [
        "hotpot_dev_distractor_v1",
        "hotpot_train_v1.1"
    ]

# Initialize the tokenizer
def get_tokenizer(model_name: str = "meta-llama/Llama-3.2-1B") -> LlamaTokenizer:
    tok = LlamaTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    return tok


def sample_wrong_paths(
    graph: nx.DiGraph,
    gold_path: List[int],
    num_wrong: int
) -> List[List[int]]:
    non_gold = [n for n in graph.nodes if n not in gold_path]
    length = len(gold_path)
    wrong_paths = []
    for _ in range(num_wrong):
        if len(non_gold) >= length:
            wrong = random.sample(non_gold, length)
        else:
            if len(non_gold) > 0:
                # fallback to allow repeats if too few non-gold nodes
                #print(non_gold, path_len)
                wrong = random.choices(non_gold, k=length)
            else:
                wrong = random.sample([n for n in graph.nodes], length)
        wrong_paths.append(wrong)
    return wrong_paths

def serialize_example(
    ex: Example,
    graph: nx.DiGraph,
    num_divergent: int = 2,
) -> tuple[str, str]:
    """
    Build a single-block Llama-2 prompt.

    • The *entire* conversation (system + user + all candidates) lives
      inside ONE  `[INST] ... [/INST]`.
    • Candidate 1 is the gold chain+answer and will be used as the *label*.
    • Candidate 2..k are divergent and get loss-masked (context only).

    Returns
    -------
    full_prompt : str   # what you feed as `text`
    gold_only   : str   # what you feed as `text_target`
    """

    lines: list[str] = []

    # ── open INST + system ─────────────────────────
    lines.append("<s>[INST] <<SYS>>")
    lines.append("You are a careful multi-hop reasoner. Think step-by-step.")
    lines.append("<</SYS>>\n")                     # blank line terminates system

    # ── user content: question + full paragraphs ──
    lines.append(f"QUESTION: {ex.question}")
    lines.append("PARAGRAPHS:")
    for pid, data in graph.nodes(data=True):
        title      = data.get("title", "")
        sentences  = " ".join(data.get("sentences", []))  # join all sentences
        # include both title and body
        lines.append(f"  {pid}: {title} || {sentences}")
    lines.append("")                                    # blank line before candidates

    # ── Candidate-1 (gold)  ───────────────────────
    lines.append("CANDIDATE 1:")
    gold_lines: list[str] = []           # will become the label slice
    for i, pid in enumerate(ex.gold_path, start=1):
        gold_lines.append(f"[STEP {i}] {pid}")
    gold_lines.append(f"FINAL: {ex.answer}")
    lines.extend("  " + ln for ln in gold_lines)   # indent in prompt

    # ── Divergent candidates  ─────────────────────
    wrong_paths = sample_wrong_paths(graph, ex.gold_path, num_divergent)
    for idx, path in enumerate(wrong_paths, start=2):
        lines.append(f"\nCANDIDATE {idx}:")
        for i, pid in enumerate(path, start=1):
            lines.append(f"  [STEP {i}] {pid}")
        lines.append("  FINAL:")

    # ── close INST and EOS ────────────────────────
    lines.append("[/INST]</s>")

    full_prompt = "\n".join(lines)
    gold_text   = "\n".join(gold_lines)

    return full_prompt, gold_text


# Pack prompt into token IDs
def pack_prompt(prompt: str, tokenizer: LlamaTokenizer, max_seq_len: int = 2048) -> dict:
    tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=max_seq_len,
        padding="max_length",
        return_tensors="pt"
    )
    return tokens

# Example usage
if __name__ == "__main__":
    project_root = get_project_root()
    processed_dir = os.path.join(project_root, "data", "processed")
    # Serialize first example
    examples = load_examples(processed_dir, get_processed_dirs()[0])
    import graph_builder
    G = graph_builder.build_graph(examples[0])
    prompt = serialize_example(examples[0], G)
    tok = get_tokenizer()
    batch = pack_prompt(prompt, tok)
    print(batch)
