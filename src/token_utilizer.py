import os
from pathlib import Path
from typing import Tuple
import networkx as nx
from transformers import LlamaTokenizer
from data_module import load_examples, Example

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


# Serialize an example and its graph into ChatML prompt for SFT
def serialize_example(ex: Example, graph: nx.DiGraph) -> str:
    """
    Builds a ChatML prompt with system, user question, gold paths, and
    placeholders for candidates (for Divergent GoT SFT).
    """
    system = "<|system|> You are a careful multi-hop reasoner. Think step-by-step."
    # Build user block
    user = ["<|user|>", f"QUESTION: {ex.question}"]
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
    user.append("CANDIDATE 1:\n  [STEP 1] ...\n  FINAL: ...")
    user.append("CANDIDATE 2:\n  [STEP 1] ...\n  FINAL: ...")
    prompt = "\n".join([system] + user + ["<|assistant|>"])
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
