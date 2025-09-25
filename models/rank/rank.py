import functools
import os
import re

import numpy as np
import pandas as pd
import torch
import tqdm
import transformers


class RankModel:
    def __init__(self, base_model_name, log, cache_dir):
        self.base_model_name = base_model_name
        self.log = log
        self.cache_dir = cache_dir
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.base_model_name, cache_dir=self.cache_dir, device_map="auto"
        )
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.base_model_name, cache_dir=self.cache_dir
        )
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        if self.base_model_name == "facebook/opt-125m":
            self.base_tokenizer.model_max_length = 2048
        elif "Llama-3" in self.base_model_name:
            self.base_tokenizer.model_max_length = 4096
        elif "Llama-2" in self.base_model_name:
            self.base_tokenizer.model_max_length = 512
        elif "mpt" in self.base_model_name:
            self.base_tokenizer.model_max_length = 512

    # Get the log likelihood of each text under the base_model
    def detect(self, text):
        if len(text) == 0:
            return 0.0
        with torch.no_grad():
            first_device = next(self.base_model.parameters()).device
            tokenized = self.base_tokenizer(
                text, truncation=True, return_tensors="pt"
            ).to(first_device)
            logits = self.base_model(**tokenized).logits[:, :-1]
            labels = tokenized.input_ids[:, 1:]
            matches = (
                logits.argsort(-1, descending=True) == labels.unsqueeze(-1)
            ).nonzero()
            ranks, _ = matches[:, -1], matches[:, -2]
            ranks = ranks.float() + 1  # convert to 1-indexed rank
            if self.log:
                ranks = torch.log(ranks)
            return ranks.float().mean().item()


class Rank:
    def __init__(self, base_model_name: str, log: bool):
        self.base_model_name = base_model_name
        self.rank_instance = RankModel(
            self.base_model_name, log, cache_dir=os.environ["HF_HOME"]
        )

    def inference(self, texts: list) -> list:
        predictions = []
        for text in tqdm.tqdm(texts):
            pred_score = -self.rank_instance.detect(text)
            predictions.append(pred_score)

        return predictions
