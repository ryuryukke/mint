from datasets import load_dataset
from itertools import islice
import pandas as pd
import pickle

pile_train = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
pile_train = pile_train.shuffle(buffer_size=10000, seed=42)
pile_train = list(islice(pile_train, 10000))
pile_test = load_dataset(
    "json",
    data_files={
        "test": "https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main/test.jsonl.zst"
    },
    split="test",
)

pile_train_df, pile_test_df = pd.DataFrame(pile_train), pile_test.to_pandas()

with open("./pile_train.pkl", "rb") as f:
    pile_train_df = pickle.load(f)
print("end of pile_train_df.to_pickle")

with open("./pile_test.pkl", "rb") as f:
    pile_test_df = pickle.load(f)
print("end of pile_test_df.to_pickle")
