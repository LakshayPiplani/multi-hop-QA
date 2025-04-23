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
def get_tokenizer(model_name: str = "meta-llama/Llama-2-7b-chat-hf") -> LlamaTokenizer:
    tok = LlamaTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    return tok


def sample_divergent(
    graph: nx.DiGraph,
    gold_path: List[int],
    num_candidates: int = 2
) -> List[List[int]]:
    """
    Sample `num_candidates` wrong hop sequences of the same length as gold_path
    by randomly selecting non-gold node IDs.
    """
    nodes = list(graph.nodes)
    non_gold = [n for n in nodes if n not in gold_path]
    path_len = len(gold_path)
    wrong_paths = []
    for i in range(num_candidates):
        if len(non_gold) >= path_len:
            if i == 0:
                wrong = [gold_path[0]]
                wrong += random.sample(non_gold, path_len-1)
            else:
                wrong = random.sample(non_gold, path_len)
        else:
            if len(non_gold) > 0:
                # fallback to allow repeats if too few non-gold nodes
                #print(non_gold, path_len)
                wrong = random.choices(non_gold, k=path_len)
            else:
                continue
        wrong_paths.append(wrong)
    return wrong_paths


# Serialize an example and its graph into ChatML prompt for SFT
def serialize_example(ex: Example, graph: nx.DiGraph, num_divergent: int = 2) -> str:
    """
    Builds a ChatML prompt with system, user question, gold paths, and
    placeholders for candidates (for Divergent GoT SFT).
    """
    system = "<s>[INST] <<SYS>>\n"
    system += "You are a careful multi-hop reasoner. Think step-by-step.\n"
    system += "<</SYS>>\n\n"
    # Build user block
    system += f"QUESTION: {ex.question}\n"
    user = []
    # Gold path
    gold = " -> ".join(str(pid) for pid in ex.gold_path)
    user.append(f"GOLD_PATH: {gold}")
    # List nodes and edges
    user.append("NODES:")
    for pid, data in graph.nodes(data=True):
        title = data.get("title", "")
        user.append(f"  {pid}: {title}")
    user.append("EDGES:")
    for u, v, attr in graph.edges(data=True):
        etype = attr.get("type", "entail")
        user.append(f"  {u} -> {v} ({etype})")
    # Placeholder for candidates

    # Candidate 1: gold path
    user.append("[INST]\nCANDIDATE 1:")
    for i, pid in enumerate(ex.gold_path, start=1):
        user.append(f"[STEP {i}] {pid}")
    user.append(f"FINAL: {ex.answer}")

    for idx in range(2, 2+num_divergent):
        path = sample_divergent(graph = graph, gold_path = ex.gold_path, num_candidates=2)
        user.append(f"CANDIDATE {idx}:")
        for i, pid in enumerate(path, start=1):
            user.append(f"[STEP {i}] {pid}")
        user.append("FINAL:")
        user.append("[/INST]</s>")
    prompt = "\n".join([system] + user)
    return prompt

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
