import os
import json
import gzip
import pickle
from tqdm import tqdm
import transformers
import numpy as np
from src.config import MODEL_MAX_LENGTH

# using the defaults of the original implementation
file_num = 15
max_tok = 1024

model2short = {
    "openai-community/gpt2-xl": "gpt2",
    "mosaicml/mpt-30b-chat": "mpt",
    "meta-llama/Llama-2-70b-chat-hf": "llama2",
    "EleutherAI/pythia-160m": "pythia",
    "meta-llama/Llama-3.2-3B": "llama3",
}


def collect_c4_texts():
    all_texts = []
    if os.path.exists(f"./c4_texts_0_{file_num - 1}.pkl"):
        with open(f"./c4_texts_0_{file_num - 1}.pkl", "rb") as f:
            all_texts = pickle.load(f)
    else:
        for i in range(file_num):
            file_name = f"c4-train.{str(i).zfill(5)}-of-01024.json.gz"
            with gzip.open(f"./{file_name}", "rt", encoding="utf-8") as f:
                for line in tqdm(f, desc=f"Reading {file_name}"):
                    example = json.loads(line)
                    all_texts.append(example["text"])
        with open(f"./c4_texts_0_{file_num - 1}.pkl", "wb") as f:
            pickle.dump(all_texts, f)
    return all_texts


def build_freq_dist(all_texts, base_tokenizer, batch_size=1024):
    freq_dist = [0] * len(base_tokenizer)

    for i in tqdm(range(0, len(all_texts), batch_size), desc="Tokenizing batches"):
        batch = all_texts[i : i + batch_size]
        outputs = base_tokenizer(batch, max_length=max_tok, truncation=True)

        for input_ids in outputs["input_ids"]:
            for token_id in input_ids:
                if token_id < len(freq_dist):
                    freq_dist[token_id] += 1

    return freq_dist


if __name__ == "__main__":
    all_texts = collect_c4_texts()

    for base_model_name in model2short.keys():
        print(f"Building a token distribution of C4 under the model: {base_model_name}")
        base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_model_name, cache_dir=os.environ["HF_HOME"]
        )
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
        for key, max_len in MODEL_MAX_LENGTH.items():
            if key in base_model_name:
                base_tokenizer.model_max_length = max_len
                break

        freq_dist = build_freq_dist(all_texts, base_tokenizer)

        with open(f"./freq_dist_{model2short[base_model_name]}.pkl", "wb") as f:
            pickle.dump(freq_dist, f)
