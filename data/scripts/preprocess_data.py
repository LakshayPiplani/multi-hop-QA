import os, argparse, json
from datasets import Dataset # type: ignore
from pathlib import Path

"""
    before running this script, please dowload the dataset by running.
       python download_data.py --output_dir <output_directory> --splits <split1> <split2> ...
"""

def preprocess(raw_dir: str, processed_dir: str, file_names: list[str]) -> None:
    """
    Load local HotpotQA JSON files (array-of-objects), normalize fields, and save as Arrow datasets.

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
        trim = 15000 if len(records) > 15000 else len(records)
        records = records[:trim]  # Limit to first 20k records for testing
        print(f"Trimming to {len(records)} records for file '{file_name}'")
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
    # find path the directory of this script
    scriptpath = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute()
    # the sub-directory raw contains the unprocessed json files of the dataset 
    raw_dir = os.path.join(scriptpath,'raw')
    # the sub-directory processed contains the processed dataset
    # the processed dataset is saved in the form of arrow files
    processed_dir = os.path.join(scriptpath,'processed')
    print(f"Raw directory: {raw_dir}")
    print(f"Processed directory: {processed_dir}")
    # take arguments from the command line
    # make list of all files in the raw directory if they are .json files
    file_names = [jsonfile for jsonfile in os.listdir(raw_dir) if jsonfile.endswith('.json')]
    print(f"Found {len(file_names)} JSON files in {raw_dir}: {file_names}")
    parser = argparse.ArgumentParser(description="Preprocess HotpotQA distractor data robustly")
    parser.add_argument("--raw_dir", type=str, default=raw_dir, help="Raw Dataset Directory")
    parser.add_argument("--processed_dir", type=str, default=processed_dir, help="Processed Dataset Directory")
    parser.add_argument("--file_names", nargs="+", default=file_names, help="Files to process")
    args = parser.parse_args()
    preprocess(args.raw_dir, args.processed_dir, args.file_names)