from transformers import AutoTokenizer
from datasets import load_dataset
import requests
import time
import argparse
from tqdm import tqdm
import json
from pathlib import Path
import pandas as pd

CURRENT_DIR = Path(__file__).parent

INDEX = "v4_piletrain_llama"
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

mimir2pile = {
    "wikipedia_(en)": "Wikipedia (en)",
    "github": "Github",
    "pile_cc": "Pile-CC",
    "pubmed_central": "PubMed Central",
    "arxiv": "ArXiv",
    "dm_mathematics": "DM Mathematics",
    "hackernews": "HackerNews",
}


def compute_cnt_via_infini_gram(ngram_ids):
    payload = {
        "index": INDEX,
        "query_type": "count",
        "query_ids": ngram_ids,
    }
    res = requests.post("https://api.infini-gram.io/", json=payload).json()
    return res["count"]


def create_ngram_ids(text, n):
    ngram_ids = []
    token_ids = llama_tokenizer.encode(text, add_special_tokens=False)
    for i in range(len(token_ids) - n + 1):
        ngram_ids.append(token_ids[i : i + n])
    return ngram_ids


def is_overlap(text, n):
    ngrams_ids_in_text = create_ngram_ids(text, n)
    if len(ngrams_ids_in_text) == 0:
        return None
    total_overlap_cnt = 0
    for ngram_ids in tqdm(ngrams_ids_in_text):
        while 1:
            try:
                overlap_cnt = compute_cnt_via_infini_gram(ngram_ids)
                if overlap_cnt:
                    total_overlap_cnt += 1
                break
            except Exception as e:
                print(e)
                time.sleep(1)
                continue
    overlap_ratio = total_overlap_cnt * 100 / len(ngrams_ids_in_text)
    return overlap_ratio


def calculate_overlap_ratio(text, n):
    overlap_ratio = is_overlap(text, n)
    return overlap_ratio


def collect_nonmember_and_member(domain: str, sample_size: int) -> dict:
    new_members, new_nonmembers = [], []

    train_path, test_path = (
        CURRENT_DIR / "pile_train.pkl",
        CURRENT_DIR / "pile_test.pkl",
    )

    pile_train_df = pd.read_pickle(train_path)
    pile_test_df = pd.read_pickle(test_path)

    pile_origin_domain_name = mimir2pile[domain]
    pile_train_df = pile_train_df[
        pile_train_df["meta"] == {"pile_set_name": pile_origin_domain_name}
    ]
    pile_test_df = pile_test_df[
        pile_test_df["meta"] == {"pile_set_name": pile_origin_domain_name}
    ]

    dataset = load_dataset("iamgroot42/mimir", domain, split="ngram_13_0.2")
    members, nonmembers = dataset["member"], dataset["nonmember"]

    print("Start of member sampling")

    pile_train_df = pile_train_df.sample(frac=1, random_state=42)
    for text in pile_train_df["text"].to_list():
        if text not in members:
            new_members.append(text)
        if len(new_members) == sample_size:
            break
    print("End of member sampling")
    print("Start of nonmember sampling")

    pile_test_df = pile_test_df.sample(frac=1, random_state=42)
    for text in pile_test_df["text"].to_list():
        overlap_ratio = calculate_overlap_ratio(text, 13)
        if overlap_ratio is None:
            continue
        if text not in nonmembers and overlap_ratio <= 20.0:
            print("found")
            new_nonmembers.append(text)
        if len(new_nonmembers) == sample_size:
            break
    print("End of nonmember sampling")

    return {"positives": new_members, "negatives": new_nonmembers}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=10)

    args = parser.parse_args()

    prefix_dir = CURRENT_DIR / "Prefix"
    prefix_dir.mkdir(parents=True, exist_ok=True)
    prefix_path = prefix_dir / f"{args.domain}_prefix.json"
    if not prefix_path.exists():
        prefix = collect_nonmember_and_member(args.domain, args.sample_size)
        prefix_path.write_text(json.dumps(prefix, ensure_ascii=False, indent=2))
    else:
        print(f"Prefix for {args.domain} already exists.")
