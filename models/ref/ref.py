import functools
import os
import re

import numpy as np
import pandas as pd
import torch
import tqdm
import transformers


class ReferenceModel:
    def __init__(self, base_model_name, ref_model_name, cache_dir):
        self.base_model_name = base_model_name
        self.ref_model_name = ref_model_name
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

        self.ref_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.ref_model_name, cache_dir=self.cache_dir, device_map="auto"
        )
        self.ref_tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.ref_model_name, cache_dir=self.cache_dir
        )
        self.ref_tokenizer.pad_token_id = self.ref_tokenizer.eos_token_id
        if self.ref_model_name == "facebook/opt-125m":
            self.ref_tokenizer.model_max_length = 2048
        elif "Llama-3" in self.ref_model_name:
            self.ref_tokenizer.model_max_length = 4096
        elif "Llama-2" in self.ref_model_name:
            self.ref_tokenizer.model_max_length = 512
        elif "mpt" in self.ref_model_name:
            self.ref_tokenizer.model_max_length = 512

    # Get the log likelihood of each text under the base_model
    def detect(self, text):
        if len(text) == 0:
            return 0.0
        with torch.no_grad():
            first_device = next(self.base_model.parameters()).device
            tokenized = self.base_tokenizer(
                text, truncation=True, return_tensors="pt"
            ).to(first_device)
            labels = tokenized.input_ids
            base_loss = self.base_model(**tokenized, labels=labels).loss.item()

            first_device = next(self.ref_model.parameters()).device
            tokenized = self.ref_tokenizer(
                text, truncation=True, return_tensors="pt"
            ).to(first_device)
            labels = tokenized.input_ids
            ref_loss = self.ref_model(**tokenized, labels=labels).loss.item()

            return -(base_loss - ref_loss)


class Reference:
    def __init__(self, base_model_name: str, ref_model_name: str):
        self.base_model_name = base_model_name
        self.ref_model_name = ref_model_name
        self.ref_instance = ReferenceModel(
            self.base_model_name, self.ref_model_name, cache_dir=os.environ["HF_HOME"]
        )

    def inference(self, texts: list) -> list:
        predictions = []
        for text in tqdm.tqdm(texts):
            pred_score = self.ref_instance.detect(text)
            predictions.append(pred_score)

        return predictions
