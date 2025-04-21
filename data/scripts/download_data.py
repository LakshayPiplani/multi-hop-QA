import os
import argparse
import requests






# URLS = {
#     "train": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
#     "dev":   "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
#     # "test":  "https://raw.githubusercontent.com/hotpotqa/hotpot/master/data/distractor_test_v1.json",
# }

# def download_file(url, dest_path):
#     resp = requests.get(url, stream=True)
#     resp.raise_for_status()
#     os.makedirs(os.path.dirname(dest_path), exist_ok=True)
#     with open(dest_path, "wb") as f:
#         for chunk in resp.iter_content(chunk_size=8192):
#             f.write(chunk)

# def main(output_dir, splits):
#     for split in splits:
#         url = URLS.get(split)
#         if url is None:
#             print(f"Unknown split: {split}")
#             continue
#         dest = os.path.join(output_dir, f"hotpot_distractor_{split}.json")
#         print(f"Downloading {split} from {url} to {dest}")
#         download_file(url, dest)
#     print("Download complete.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Download HotpotQA distractor data")
#     parser.add_argument("--output_dir", type=str, default="data/raw", help="Directory to save raw JSON files")
#     parser.add_argument("--splits",    type=str, nargs="+", default=["train","dev"], help="Which splits to download")
#     args = parser.parse_args()
#     main(args.output_dir, args.splits)
