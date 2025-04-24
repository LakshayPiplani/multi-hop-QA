import os
from typing import List
import networkx as nx
from data_module import load_examples, Example, Node


def build_graph(example: Example) -> nx.DiGraph:
    """
    Build a directed graph for one multi-hop QA example.

    Nodes:
      pid (int) with attributes: title, sentences, is_gold
    Edges:
      - "entail" between consecutive gold_path pids
      - "lexical" if paragraphs share ≥3 tokens
      - "title_link" if one paragraph's title appears in another's text

    Returns:
        A networkx.DiGraph object.
    """
    G = nx.DiGraph()
    # Add nodes
    for node in example.nodes:
        G.add_node(
            node.pid,
            title=node.title,
            sentences=node.sentences,
            is_gold=node.is_gold
        )

    # Add gold entailment edges
    for u, v in zip(example.gold_path, example.gold_path[1:]):
        G.add_edge(u, v, type="entail")

    # Precompute paragraph token sets for lexical edges
    token_sets = {}
    for node in example.nodes:
        text = " ".join(node.sentences).lower()
        tokens = set(t.strip('.,!?;') for t in text.split())
        token_sets[node.pid] = tokens

    # Add lexical and title_link edges
    for u in example.nodes:
        for v in example.nodes:
            if u.pid == v.pid:
                continue
            # Lexical overlap
            overlap = token_sets[u.pid] & token_sets[v.pid]
            if len(overlap) >= 3:
                G.add_edge(u.pid, v.pid, type="lexical")
            # Title link
            title_phrase = u.title.replace("_", " ").lower()
            if title_phrase in " ".join(v.sentences).lower():
                G.add_edge(u.pid, v.pid, type="title_link")

    return G


def main():
    # Determine project root and processed data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    processed_dir = os.path.join(project_root, "data", "processed")


    # finding list of subdirectories in processed_dir
    sub_dirs = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
    print(f"Subdirectories in {processed_dir}: {sub_dirs}")
    # Define your processed subdirectories
    sub_dirs = ["hotpot_dev_distractor_v1", "hotpot_train_v1.1"]

    for sub_dir in sub_dirs:
        print(f"Loading examples from {sub_dir}...")
        examples: List[Example] = load_examples(processed_dir, sub_dir)
        print(f"Building graphs for {len(examples)} examples...")
        graphs = [build_graph(ex) for ex in examples]
        print(f"Built {len(graphs)} graphs for {sub_dir}\n")

    # Optionally: return graphs or save them for downstream use

if __name__ == "__main__":
    main()