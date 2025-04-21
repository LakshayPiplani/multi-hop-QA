import os
import json
import argparse
from pathlib import Path

# Assumes you have these modules implemented:
# from retriever import Retriever
# from infer import HotpotInferencer

def load_test_data(input_file):
    """
    Load test examples. For distractor mode, data contains 'context' field.
    For fullwiki mode, data contains only 'question'.
    """
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data

def main():
    parser = argparse.ArgumentParser(description="Generate HotpotQA predictions for CodaLab submission")
    parser.add_argument("--input_file",  type=str, required=True,
                        help="Path to test JSON file (fullwiki or distractor)")
    parser.add_argument("--model_dir",    type=str, required=True,
                        help="Directory containing the trained Hotpot inferencer")
    parser.add_argument("--output_file",  type=str, default="submission.json",
                        help="Output JSON file for CodaLab submission")
    parser.add_argument("--mode",         type=str, choices=["fullwiki","distractor"], default="fullwiki",
                        help="Evaluation mode: 'fullwiki' (run retriever) or 'distractor' (use given contexts)")
    parser.add_argument("--retriever_index", type=str, default="",
                        help="Path to FAISS index (required for fullwiki)")
    args = parser.parse_args()

    # Initialize inferencer
    inferencer = HotpotInferencer(model_dir=args.model_dir)

    # Initialize retriever if needed
    retriever = None
    if args.mode == "fullwiki":
        if not args.retriever_index:
            raise ValueError("Fullwiki mode requires --retriever_index")
        retriever = Retriever(index_path=args.retriever_index)

    # Load test data
    test_data = load_test_data(args.input_file)

    answers = {}
    sp     = {}

    for item in test_data:
        qid      = item.get("qid") or item.get("_id")
        question = item["question"]
        if args.mode == "distractor":
            context = item["context"]
        else:
            # retrieve top-k paragraphs
            context = retriever.retrieve(question, topk=10)

        # Run inference: return (answer_str, list_of_[title, sent_id] pairs)
        answer_str, support_facts = inferencer.predict(question, context)

        answers[qid] = answer_str
        sp[qid]      = support_facts

    # Build submission dict
    submission = {"answer": answers, "sp": sp}

    # Save to JSON
    with open(args.output_file, "w") as f:
        json.dump(submission, f, indent=2)

    print(f"Saved submission file: {args.output_file}")

if __name__ == "__main__":
    main()
'''

os.makedirs("src", exist_ok=True)
with open("src/predict_codalab.py", "w") as f:
    f.write(predict_codalab_code)

print("=== src/predict_codalab.py ===\n")
print(predict_codalab_code)
