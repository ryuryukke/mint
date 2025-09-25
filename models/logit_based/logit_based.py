import functools
import os
import re

import numpy as np
import pandas as pd
import torch
import tqdm
import transformers
import zlib
import torch.nn.functional as F


class LogitBasedModel:
    def __init__(self, base_model_name, ref_model_name, methods, cache_dir):
        self.base_model_name = base_model_name
        self.ref_model_name = ref_model_name
        self.methods = methods
        self.cache_dir = cache_dir

        max_mem = {i: "46GiB" for i in range(torch.cuda.device_count() - 1)} # todo: adjust memory
        last = torch.cuda.device_count() - 1
        
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.base_model_name, trust_remote_code=True, cache_dir=self.cache_dir, device_map="auto", max_memory=max_mem
        )
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.base_model_name, trust_remote_code=True, cache_dir=self.cache_dir
        )
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        # self.base_tokenizer.model_max_length = 512
        if self.base_model_name == "facebook/opt-125m":
            self.base_tokenizer.model_max_length = 2048
        elif "Llama-3" in self.base_model_name:
            self.base_tokenizer.model_max_length = 4096
        elif "Llama-2" in self.base_model_name:
            self.base_tokenizer.model_max_length = 512
        elif "mpt" in self.base_model_name:
            self.base_tokenizer.model_max_length = 512
        elif "gpt2" in self.base_model_name:
            self.base_tokenizer.model_max_length = 1024
        elif "pythia" in self.base_model_name:
            self.base_tokenizer.model_max_length = 2048
        

        self.ref_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.ref_model_name, trust_remote_code=True, cache_dir=self.cache_dir, device_map={"": last},
        )
        self.ref_tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.ref_model_name, trust_remote_code=True, cache_dir=self.cache_dir
        )
        self.ref_tokenizer.pad_token_id = self.ref_tokenizer.eos_token_id
        # self.ref_tokenizer.model_max_length = 512
        if self.ref_model_name == "facebook/opt-125m":
            self.ref_tokenizer.model_max_length = 2048
        elif "Llama-3" in self.ref_model_name:
            self.ref_tokenizer.model_max_length = 4096
        elif "Llama-2" in self.ref_model_name:
            self.ref_tokenizer.model_max_length = 512
        elif "mpt" in self.ref_model_name:
            self.ref_tokenizer.model_max_length = 512
        elif "gpt2" in self.ref_model_name:
            self.ref_tokenizer.model_max_length = 1024
        elif "pythia" in self.ref_model_name:
            self.ref_tokenizer.model_max_length = 2048
        elif "stablelm" in self.ref_model_name:
            self.ref_tokenizer.model_max_length = 4096
        

    # Get the log likelihood of each text under the base_model
    def detect(self, text):
        predictions = dict((method, 0.0) for method in self.methods)
        if len(text) == 0:
            return predictions
        with torch.no_grad():
            first_device = next(self.base_model.parameters()).device
            input_ids = torch.tensor(self.base_tokenizer.encode(text, truncation=True)).unsqueeze(0)
            input_ids = input_ids.to(first_device)
            with torch.no_grad():
                outputs = self.base_model(input_ids, labels=input_ids)
            base_loss, logits = outputs[:2]

            # loss
            predictions['loss'] = -base_loss.item()
            
            # zlib
            zlib_entropy = len(zlib.compress(bytes(text, "utf-8")))
            predictions['zlib'] = -base_loss.item() / zlib_entropy
            
            # entropy
            logits_ = logits[:, :-1]
            neg_entropy = F.softmax(logits_, dim=-1) * F.log_softmax(logits_, dim=-1)
            predictions['entropy'] = neg_entropy.sum(-1).mean().item()
            
            # min_k & min_k_plus
            input_ids_ = input_ids[0][1:].unsqueeze(-1)
            probs = F.softmax(logits[0, :-1], dim=-1)
            log_probs = F.log_softmax(logits[0, :-1], dim=-1)
            token_log_probs = log_probs.gather(dim=-1, index=input_ids_).squeeze(-1)
            mu = (probs * log_probs).sum(-1)
            sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
            ratio = 0.2
            # min_k
            k_length = int(len(token_log_probs) * ratio)
            topk = np.sort(token_log_probs.cpu())[:k_length]
            predictions['min_k'] = np.mean(topk).item()
            # min_k_plus
            mink_plus = (token_log_probs - mu) / sigma.sqrt()
            k_length = int(len(mink_plus) * ratio)
            topk = np.sort(mink_plus.cpu())[:k_length]
            predictions['min_k_plus'] = np.mean(topk).item()
            
            # rank & logrank
            logits_ = logits[:, :-1]
            labels = input_ids[:, 1:]
            matches = (
                logits_.argsort(-1, descending=True) == labels.unsqueeze(-1)
            ).nonzero()
            ranks, _ = matches[:, -1], matches[:, -2]
            ranks = ranks.float() + 1  # convert to 1-indexed rank
            logranks = torch.log(ranks)
            predictions['rank'] = -ranks.float().mean().item()
            predictions['logrank'] = -logranks.float().mean().item()
            
            # ref
            first_device = next(self.ref_model.parameters()).device
            input_ids = torch.tensor(self.ref_tokenizer.encode(text, truncation=True)).unsqueeze(0)
            input_ids = input_ids.to(first_device)
            with torch.no_grad():
                outputs = self.ref_model(input_ids, labels=input_ids)
            ref_loss, _ = outputs[:2]
            predictions['ref'] = -(base_loss.item() - ref_loss.item())

        return predictions


class LogitBased:
    def __init__(self, base_model_name: str, ref_model_name: str):
        self.base_model_name = base_model_name
        self.ref_model_name = ref_model_name
        self.methods = ["loss", "ref", "zlib", "min_k", "min_k_plus", "rank", "logrank", "entropy"]
        self.logit_based_instance = LogitBasedModel(
            self.base_model_name, self.ref_model_name, self.methods, cache_dir=os.environ["HF_HOME"]
        )
        

    def inference(self, texts: list) -> list:
        predictions = dict((method, []) for method in self.methods)
        for text in tqdm.tqdm(texts):
            pred_score = self.logit_based_instance.detect(text)
            for method in self.methods:
                predictions[method].append(pred_score[method])
        return predictions
