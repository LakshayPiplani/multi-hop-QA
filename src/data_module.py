import os
from dataclasses import dataclass
from typing import List, Optional
from datasets import load_from_disk, Dataset


@dataclass
class Node:
    pid: int
    title: str
    sentences: List[str]
    is_gold: bool

@dataclass
class Example:
    qid: str
    question: str
    answer: Optional[str]
    nodes: List[Node]
    gold_path: List[int]


def load_examples(processed_dir: str, sub_dir: str) -> List[Example]:
    """
    Load processed HotpotQA Arrow dataset for a given sub directory and return a list of Example objects.

    Args:
        processed_dir: directory where Arrow datasets are saved (e.g. data/processed).
        sub_sir: name of sub_directory in which Arrow files live

    Returns:
        List of Example dataclass instances.
    """
    path = os.path.join(processed_dir, sub_dir)
    ds: Dataset = load_from_disk(path)

    examples: List[Example] = []
    for record in ds.select(range(500)):
        qid = record.get("_id", "")
        question = record["question"]
        answer = record.get("answer")  # may be None for test

        # Build mapping from (title, sent_id) to pid
        sf = record.get("supporting_facts", [])
        # Convert list of dicts to tuples
        sf_tuples = [(item['title'], item['sent_id']) for item in sf]
        # Build pid mapping for gold_path
        gold_path = []
        
        # Build nodes list
        nodes: List[Node] = []
        context = record["context"]  # list of dicts {'title':..., 'sentences':[...]}
        for pid, para in enumerate(context):
            title = para['title']
            sentences = para['sentences']
            # Determine if this paragraph contains any supporting fact for gold_path
            is_gold = False
            for (t, idx) in sf_tuples:
                if t == title:
                    # record the pid for each matching supporting fact in order
                    gold_path.append(pid)
                    is_gold = True
            nodes.append(Node(pid=pid, title=title, sentences=sentences, is_gold=is_gold))

        # Deduplicate gold_path while preserving order
        seen = set()
        ordered_path = []
        for pid in gold_path:
            if pid not in seen:
                seen.add(pid)
                ordered_path.append(pid)

        examples.append(Example(qid=qid, question=question, answer=answer,
                                nodes=nodes, gold_path=ordered_path))
    return examples

#if __name__ == "__main__":
    # Usage example:
    #script_dir = os.path.dirname(os.path.abspath(__file__))
    #processed_dir = os.path.join(script_dir, '..', 'data', 'processed')
    #examples = load_examples(processed_dir=processed_dir, sub_dir="hotpot_dev_distractor_v1")
    #print(f"Loaded {len(examples)} examples")
