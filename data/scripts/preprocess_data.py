import os
import argparse
import json
from datasets import Dataset


def preprocess(raw_dir: str, processed_dir: str, file_names: list[str]) -> None:
    """
    Load local HotpotQA JSON files (array-of-objects), normalize fields,
    and save as Arrow datasets.

    Each JSON file is a list of dicts with keys:
      _id: str
      question: str
      answer: str (absent in test)
      supporting_facts: list of [title, sent_id] (absent in test)
      context: list of [title, sentences], sentences is list of str

    This function normalizes:
      - supporting_facts -> list of dict{'title': str, 'sent_id': int}
      - context -> list of dict{'title': str, 'sentences': List[str]}
    """
    os.makedirs(processed_dir, exist_ok=True)

    for file_name in file_names:
        print(f"\nStarting processing for file '{file_name}'")
        json_path = os.path.join(raw_dir, file_name)
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Could not find JSON at {json_path}")

        print(f"Reading raw JSON from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            records = json.load(f)

        if not isinstance(records, list):
            raise ValueError(f"Expected top-level list in JSON, got {type(records)}")

        print(f"Loaded {len(records)} records; normalizing fields...")
        normalized = []
        for rec in records:
            rec_norm = {}
            # Copy primitive fields
            rec_norm['_id'] = rec.get('_id')
            rec_norm['question'] = rec.get('question')
            if 'answer' in rec:
                rec_norm['answer'] = rec['answer']

            # Normalize supporting_facts
            sf = rec.get('supporting_facts', [])
            if isinstance(sf, list):
                rec_norm['supporting_facts'] = [
                    {'title': pair[0], 'sent_id': int(pair[1])}
                    for pair in sf
                    if isinstance(pair, (list, tuple)) and len(pair) == 2
                ]
            else:
                rec_norm['supporting_facts'] = []

            # Normalize context
            ctx = rec.get('context', [])
            if isinstance(ctx, list):
                rec_norm['context'] = [
                    {'title': item[0], 'sentences': item[1]}
                    for item in ctx
                    if isinstance(item, list) and len(item) == 2
                ]
            else:
                rec_norm['context'] = []

            normalized.append(rec_norm)

        # Create Hugging Face Dataset
        ds = Dataset.from_list(normalized)

        # Save to Arrow format
        base = os.path.splitext(file_name)[0]
        out_dir = os.path.join(processed_dir, base)
        print(f"Saving processed dataset to {out_dir}...")
        ds.save_to_disk(out_dir)

    print("Preprocessing complete.")


if __name__ == "__main__":
    # fsspec.config.conf['default_block_size'] = 20 * 2**20  # 20 MiB
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(script_dir, '..', 'raw')
    processed_dir = os.path.join(script_dir, '..', 'processed')

    parser = argparse.ArgumentParser(description="Preprocess HotpotQA distractor data robustly")
    parser.add_argument("--raw_dir", type=str, default=raw_dir,
                        help="Directory to rea raw JSON files")
    parser.add_argument("--processed_dir", type=str, default=processed_dir,
                        help="Directory to save processed datasets")
    parser.add_argument("--file_names", nargs="+", default=["hotpot_dev_distractor_v1.json", "hotpot_train_v1.1.json"],
                        help="Which files to process")
    args = parser.parse_args()
    preprocess(args.raw_dir, args.processed_dir, args.file_names)
