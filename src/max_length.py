import os
import argparse
from pathlib import Path
from datasets import load_from_disk
from transformers import LlamaTokenizer

from huggingface_hub import login
login(token="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")

def analyze_paragraph_lengths(processed_dir: str, sub_dirs: list[str], use_tokenizer: bool = True):
    """
    Analyzes paragraph lengths in the processed HotpotQA datasets.

    Args:
        processed_dir: Root directory where sub_dirs live (each sub_dir contains an Arrow dataset).
        sub_dirs: List of sub-directory names to analyze (e.g. ['hotpot_dev_distractor_v1', 'hotpot_train_v1.1']).
        use_tokenizer: If True, measure lengths in tokens; otherwise, measure word counts.
    """
    # Initialize tokenizer if needed
    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf') if use_tokenizer else None

    stats = {}
    for sub in sub_dirs:
        ds_path = os.path.join(processed_dir, sub)
        if not os.path.isdir(ds_path):
            print(f"Warning: Dataset directory not found: {ds_path}, skipping.")
            continue
        print(f"Loading dataset from {ds_path}...")
        ds = load_from_disk(ds_path)

        max_len = 0
        total_len = 0
        count = 0

        for record in ds:
            # record['context'] is a list of dicts {'title':..., 'sentences':[...]}
            for para in record['context']:
                text = ' '.join(para['sentences'])
                if use_tokenizer:
                    tokens = tokenizer.encode(text)
                    length = len(tokens)
                else:
                    length = len(text.split())
                max_len = max(max_len, length)
                total_len += length
                count += 1

        avg_len = total_len / count if count > 0 else 0
        stats[sub] = {'max_length': max_len, 'avg_length': avg_len, 'num_paragraphs': count}

    # Print summary
    print("\nParagraph Length Analysis")
    for sub, s in stats.items():
        unit = 'tokens' if use_tokenizer else 'words'
        print(f"Dataset {sub} -> max {s['max_length']} {unit}, avg {s['avg_length']:.1f} {unit}, paragraphs {s['num_paragraphs']}")


def main():
    parser = argparse.ArgumentParser(description="Analyze paragraph lengths in processed HotpotQA data")
    parser.add_argument(
        "--processed_dir", type=str,
        default=os.path.join(Path(__file__).parent.parent, 'data', 'processed'),
        help="Directory containing processed datasets"
    )
    parser.add_argument(
        "--sub_dirs", nargs='+', default=['hotpot_dev_distractor_v1', 'hotpot_train_v1.1'],
        help="List of sub-directories under processed_dir to analyze"
    )
    parser.add_argument(
        "--no_tokenizer", action='store_true',
        help="If set, measure paragraph length in words instead of tokens"
    )
    args = parser.parse_args()

    analyze_paragraph_lengths(
        processed_dir=args.processed_dir,
        sub_dirs=args.sub_dirs,
        use_tokenizer=not args.no_tokenizer
    )

if __name__ == '__main__':
    main()
