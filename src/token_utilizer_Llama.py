import os
from pathlib import Path
from typing import Tuple, List
import networkx as nx
from transformers import LlamaTokenizer
from data_module import load_examples, Example
import random

MODEL_ID = "meta-llama/Llama-3.2-1B"

# Compute project root based on this file's location
def get_project_root() -> Path:
    return Path(__file__).parent.parent.resolve()



# Initialize the tokenizer
def get_tokenizer(model_name: str = MODEL_ID) -> LlamaTokenizer:
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
    Llama-3 style prompt.

    ┌─ system header ──────────────────────────────────────────┐
    | <|start_header_id|>system<|end_header_id|>               |
    |  …instruction…                                           |
    | <|eot_id|>                                               |
    └──────────────────────────────────────────────────────────┘
    ┌─ user header ────────────────────────────────────────────┐
    | <|start_header_id|>user<|end_header_id|>                 |
    |  QUESTION …                                             |
    |  PARAGRAPHS:                                            |
    |    pid: title || sentences                              |
    |                                                         |
    |  CANDIDATE 1:   ← gold chain                            |
    |    [STEP 1] pid …                                       |
    |    FINAL: answer                                        |
    |                                                         |
    |  CANDIDATE 2:   ← divergent (context-only)              |
    |  …                                                     |
    | <|eot_id|>                                              |
    └──────────────────────────────────────────────────────────┘
    ┌─ assistant header (label) ───────────────────────────────┐
    | <|start_header_id|>assistant<|end_header_id|>            |
    |  (gold chain + answer)   ← tokens that get loss         |
    └──────────────────────────────────────────────────────────┘
    """

    # ── system block ───────────────────────────────────────
    system = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are a careful multi-hop reasoner. Think step-by-step."
        "<|eot_id|>"
    )

    # ── user block: question + paragraphs + candidates ─────
    user_lines = [
        "<|start_header_id|>user<|end_header_id|>",
        f"QUESTION: {ex.question}",
        "PARAGRAPHS:",
    ]
    for pid, data in graph.nodes(data=True):
        title = data.get("title", "")
        sent  = " ".join(data.get("sentences", []))
        user_lines.append(f"  {pid}: {title} || {sent}")

    # Candidate-1 (gold) lines (also captured for labels)
    gold_lines = [f"[STEP {i}] {pid}"
                  for i, pid in enumerate(ex.gold_path, 1)]
    gold_lines.append(f"FINAL: {ex.answer}")

    user_lines.append("\nCANDIDATE 1:")
    user_lines.extend("  " + ln for ln in gold_lines)

    # Divergent candidates (context only)
    wrong_paths = sample_wrong_paths(graph, ex.gold_path, num_divergent)
    for idx, path in enumerate(wrong_paths, start=2):
        user_lines.append(f"\nCANDIDATE {idx}:")
        for i, pid in enumerate(path, 1):
            user_lines.append(f"  [STEP {i}] {pid}")
        user_lines.append("  FINAL:")

    user_lines.append("<|eot_id|>")
    user_block = "\n".join(user_lines)

    # ── assistant header: label text only ──────────────────
    assistant_block = (
        "<|start_header_id|>assistant<|end_header_id|>\n" +
        "\n".join(gold_lines)
    )

    full_prompt = "\n".join([system, user_block, assistant_block])
    gold_text   = "\n".join(gold_lines)          # label slice

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
# if __name__ == "__main__":
#     project_root = get_project_root()
#     processed_dir = os.path.join(project_root, "data", "processed")
#     # Serialize first example
#     examples = load_examples(processed_dir, get_processed_dirs()[0])
#     import graph_builder
#     G = graph_builder.build_graph(examples[0])
#     prompt = serialize_example(examples[0], G)
#     tok = get_tokenizer()
#     batch = pack_prompt(prompt, tok)
#     print(batch)
