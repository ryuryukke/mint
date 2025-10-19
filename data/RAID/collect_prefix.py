import pandas as pd
import os
from raid.utils import load_data
import random
from pathlib import Path

random.seed(42)

"""
Note: In our paper, the original prefixes of ReCaLL for detection were retrieved from the confidential RAID-test set.
In this public version, we instead use prefixes sampled from the RAID-train set, which is the test set in our paper.
As the test set contains over 1,000 samples, this change is expected to have a negligible effect on the overall results.
"""

CURRENT_DIR = Path(__file__).parent

domains = [
    "abstracts",
    "books",
    "news",
    "poetry",
    "recipes",
    "reddit",
    "reviews",
    "wiki",
]

generators = ["chatgpt", "gpt4", "llama-chat", "gpt2", "mpt-chat"]

print("Loading the RAID dataset...")
raid_path = CURRENT_DIR / "raid_train.pkl"
if raid_path.exists():
    train_df = pd.read_pickle(raid_path)
else:
    train_df = load_data(split="train")
    pd.to_pickle(train_df, raid_path)
print("Finished loading the RAID dataset.")

prefix_dir = CURRENT_DIR / "Prefix"
prefix_dir.mkdir(parents=True, exist_ok=True)

for domain in domains:
    for generator in generators:
        print(domain, generator)
        humans = train_df[
            (train_df["domain"] == domain)
            & (train_df["model"] == "human")
            & (train_df["attack"] == "none")
        ]
        machines = train_df[
            (train_df["domain"] == domain)
            & (train_df["model"] == generator)
            & (train_df["repetition_penalty"] == "no")
            & (train_df["attack"] == "none")
            & (train_df["decoding"] == "sampling")
        ]

        humans = humans.sample(frac=1, random_state=42)
        machines = machines.sample(frac=1, random_state=42)

        negative_sample, positive_sample = (
            humans["generation"].to_list()[:10],
            machines["generation"].to_list()[:10],
        )

        d = {"negatives": negative_sample, "positives": positive_sample}
        df_samples = pd.DataFrame(d)
        df_samples.to_json(prefix_dir / f"{domain}_{generator}_prefix.json")
