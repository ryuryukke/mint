"""
Adapted from the code of the paper "RECALL: Membership Inference via Relative Conditional Log-Likelihoods": https://arxiv.org/pdf/2406.15968
"""

import os
import numpy as np
import torch
import tqdm
import transformers
from src.config import MODEL_MAX_LENGTH


class RecallModel:
    def __init__(self, base_model_name: str, cache_dir: str):
        self.base_model_name = base_model_name
        self.cache_dir = cache_dir
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.base_model_name, cache_dir=self.cache_dir, device_map="auto"
        )
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.base_model_name, cache_dir=self.cache_dir
        )
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        for key, max_len in MODEL_MAX_LENGTH.items():
            if key in base_model_name:
                self.base_tokenizer.model_max_length = max_len
                break

    def get_conditional_ll(self, text: str, prefix: list) -> float:
        first_device = next(self.base_model.parameters()).device
        input_encodings = self.base_tokenizer(
            "".join(prefix), truncation=True, return_tensors="pt"
        ).to(first_device)
        target_encodings = self.base_tokenizer(
            text, truncation=True, return_tensors="pt"
        ).to(first_device)
        concat_ids = torch.cat(
            (input_encodings.input_ids, target_encodings.input_ids), dim=1
        )
        # to avoid out-of-index error
        max_length = (
            self.base_model.config.max_position_embeddings
            if "mpt" not in self.base_model_name
            else self.base_model.config.max_seq_len
        )
        if concat_ids.shape[1] > max_length:
            excess = concat_ids.shape[1] - max_length
            concat_ids = concat_ids[:, excess:]
            labels = concat_ids.clone()
            labels[:, : input_encodings.input_ids.size(1) - excess] = -100
        else:
            labels = concat_ids.clone()
            labels[:, : input_encodings.input_ids.size(1)] = -100

        with torch.no_grad():
            outputs = self.base_model(concat_ids, labels=labels)
        loss, _ = outputs[:2]
        return -loss.item()

    def process_prefix(self, prefix: list, avg_length: int) -> list:
        max_length = (
            self.base_model.config.max_position_embeddings
            if "mpt" not in self.base_model_name
            else self.base_model.config.max_seq_len
        )
        token_counts = [
            len(self.base_tokenizer.encode(shot, truncation=True)) for shot in prefix
        ]
        target_token_count = avg_length
        total_tokens = sum(token_counts) + target_token_count
        if total_tokens <= max_length:
            return prefix
        # Determine the maximum number of shots that can fit within the max_length
        max_shots = 0
        cumulative_tokens = target_token_count
        for count in token_counts:
            if cumulative_tokens + count <= max_length:
                max_shots += 1
                cumulative_tokens += count
            else:
                break
        # Truncate the prefix to include only the maximum number of shots
        truncated_prefix = prefix[-max_shots:]
        return truncated_prefix

    def detect(self, text: str, negative_prefix: list) -> float:
        if len(text) == 0:
            return 0.0
        with torch.no_grad():
            first_device = next(self.base_model.parameters()).device
            tokenized = self.base_tokenizer(
                text, truncation=True, return_tensors="pt"
            ).to(first_device)

            # get unconditional log likelihood
            labels = tokenized.input_ids
            ll = -self.base_model(**tokenized, labels=labels).loss.item()

            # get conditional log likelihood with prefix
            ll_negative = self.get_conditional_ll(text, negative_prefix)

            return ll_negative / ll


class Recall:
    def __init__(self, base_model_name: str, prefix: dict):
        self.base_model_name = base_model_name
        self.recall_instance = RecallModel(
            self.base_model_name, cache_dir=os.environ["HF_HOME"]
        )
        self.prefix = prefix

    def inference(self, texts: list) -> list:
        predictions = []
        # negative samples means non-member samples in MIAs and human-written samples in detection
        avg_length = int(
            np.mean(
                [
                    len(
                        self.recall_instance.base_tokenizer.encode(
                            text, truncation=True
                        )
                    )
                    for text in texts
                ]
            )
        )
        negative_prefix = self.recall_instance.process_prefix(
            self.prefix["negatives"], avg_length
        )
        for text in tqdm.tqdm(texts):
            pred_score = self.recall_instance.detect(text, negative_prefix)
            predictions.append(pred_score)

        return {"recall": predictions}
