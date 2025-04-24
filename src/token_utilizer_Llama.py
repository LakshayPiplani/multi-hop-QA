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

def serialize_example(ex: Example, graph: nx.DiGraph, num_divergent: int = 2) -> str:
    """
    Build a Llama-2 Chat prompt with one INST block containing:
      - system message
      - QUESTION + GOLD_PATH + graph structure
      - CANDIDATE 1 (gold) + CANDIDATE 2…k (divergent)
    """
    lines = []
    # Open INST + system
    lines.append("<s>[INST] <<SYS>>")
    lines.append("You are a careful multi-hop reasoner. Think step-by-step.")
    lines.append("<</SYS>>\n")  # end of system + blank line
    
    # User prompt inside the same INST
    lines.append(f"QUESTION: {ex.question}")
    gold_str = " -> ".join(str(pid) for pid in ex.gold_path)
    lines.append(f"GOLD_PATH: {gold_str}\n")
    
    # Graph structure
    lines.append("NODES:")
    for pid, data in graph.nodes(data=True):
        lines.append(f"  {pid}: {data.get('title','')}")
    lines.append("EDGES:")
    for u, v, attr in graph.edges(data=True):
        lines.append(f"  {u} -> {v} ({attr.get('type','entail')})")
    lines.append("")  # blank line before candidates
    
    # Candidate 1 (gold)
    lines.append("CANDIDATE 1:")
    for i, pid in enumerate(ex.gold_path, start=1):
        lines.append(f"  [STEP {i}] {pid}")
    lines.append(f"  FINAL: {ex.answer or ''}\n")
    
    # Sample and emit divergent candidates
    wrong_paths = sample_wrong_paths(graph, ex.gold_path, num_divergent)
    for idx, path in enumerate(wrong_paths, start=2):
        lines.append(f"CANDIDATE {idx}:")
        for i, pid in enumerate(path, start=1):
            lines.append(f"  [STEP {i}] {pid}")
        lines.append("  FINAL:\n")  # model should fill this
    
    # Close INST and EOS
    lines.append("[/INST]</s>")
    
    return "\n".join(lines)

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
