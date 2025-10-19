import os
import numpy as np
import torch
import tqdm
import transformers
import zlib
import torch.nn.functional as F
from src.config import MODEL_MAX_LENGTH


class BaselinesModel:
    def __init__(
        self, base_model_name: str, ref_model_name: str, methods: list, cache_dir: str
    ):
        self.base_model_name = base_model_name
        self.ref_model_name = ref_model_name
        self.methods = methods
        self.cache_dir = cache_dir

        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            device_map="auto",
        )
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.base_model_name, trust_remote_code=True, cache_dir=self.cache_dir
        )
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        for key, max_len in MODEL_MAX_LENGTH.items():
            if key in self.base_model_name:
                self.base_tokenizer.model_max_length = max_len
                break

        self.ref_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.ref_model_name,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            device_map="auto",
        )
        self.ref_tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.ref_model_name, trust_remote_code=True, cache_dir=self.cache_dir
        )
        self.ref_tokenizer.pad_token_id = self.ref_tokenizer.eos_token_id
        for key, max_len in MODEL_MAX_LENGTH.items():
            if key in self.ref_model_name:
                self.ref_tokenizer.model_max_length = max_len
                break

    def detect(self, text: str) -> dict:
        predictions = dict((method, 0.0) for method in self.methods)
        if len(text) == 0:
            return predictions
        with torch.no_grad():
            first_device = next(self.base_model.parameters()).device
            input_ids = torch.tensor(
                self.base_tokenizer.encode(text, truncation=True)
            ).unsqueeze(0)
            input_ids = input_ids.to(first_device)
            with torch.no_grad():
                outputs = self.base_model(input_ids, labels=input_ids)
            base_loss, logits = outputs[:2]

            # loss
            predictions["loss"] = -base_loss.item()

            # entropy
            logits_ = logits[:, :-1]
            neg_entropy = F.softmax(logits_, dim=-1) * F.log_softmax(logits_, dim=-1)
            predictions["entropy"] = neg_entropy.sum(-1).mean().item()

            # rank & logrank
            logits_ = logits[:, :-1]
            labels = input_ids[:, 1:]
            matches = (
                logits_.argsort(-1, descending=True) == labels.unsqueeze(-1)
            ).nonzero()
            ranks, _ = matches[:, -1], matches[:, -2]
            ranks = ranks.float() + 1
            logranks = torch.log(ranks)
            predictions["rank"] = -ranks.float().mean().item()
            predictions["logrank"] = -logranks.float().mean().item()

        return predictions


class Baselines:
    def __init__(self, base_model_name: str, ref_model_name: str):
        self.base_model_name = base_model_name
        self.ref_model_name = ref_model_name
        self.methods = ["loss", "entropy", "rank", "logrank"]
        self.baselines_instance = BaselinesModel(
            self.base_model_name,
            self.ref_model_name,
            self.methods,
            cache_dir=os.environ["HF_HOME"],
        )

    def inference(self, texts: list) -> dict:
        predictions = dict((method, []) for method in self.methods)
        for text in tqdm.tqdm(texts):
            pred_score = self.baselines_instance.detect(text)
            for method in self.methods:
                predictions[method].append(pred_score[method])
        return predictions
