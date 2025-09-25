import functools
import os
import re

import numpy as np
import pandas as pd
import torch
import tqdm
import transformers
import torch.nn.functional as F


class MinKModel:
    def __init__(self, base_model_name, cache_dir):
        self.base_model_name = base_model_name
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

    def detect(self, text):
        if len(text) == 0:
            return 0.0
        first_device = next(self.base_model.parameters()).device
        input_ids = torch.tensor(
            self.base_tokenizer.encode(text, truncation=True)
        ).unsqueeze(0)
        input_ids = input_ids.to(first_device)
        with torch.no_grad():
            outputs = self.base_model(input_ids, labels=input_ids)
        _, logits = outputs[:2]

        input_ids = input_ids[0][1:].unsqueeze(-1)
        log_probs = F.log_softmax(logits[0, :-1], dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)

        ratio = 0.2
        k_length = int(len(token_log_probs) * ratio)
        topk = np.sort(token_log_probs.cpu())[:k_length]
        return np.mean(topk).item()


class MinK:
    def __init__(self, base_model_name: str):
        self.base_model_name = base_model_name
        self.min_k_instance = MinKModel(
            self.base_model_name, cache_dir=os.environ["HF_HOME"]
        )

    def inference(self, texts: list) -> list:
        predictions = []
        for text in tqdm.tqdm(texts):
            pred_score = self.min_k_instance.detect(text)
            predictions.append(pred_score)

        return predictions
