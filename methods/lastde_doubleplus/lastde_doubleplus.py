"""
Adapted from the original implementation of Lastde: https://github.com/TrustMedia-zju/Lastde_Detector
Some modifications added.
"""

import os
import numpy as np
import torch
import tqdm
import transformers
from methods.lastde_doubleplus import fastMDE
import random
from src.config import MODEL_MAX_LENGTH


class LastdeDoublePlusModel:
    def __init__(
        self,
        n_samples=100,
        base_model_name="",
        embed_size=4,
        epsilon=8,
        tau_prime=15,
        seed=0,
        cache_dir=os.environ["HF_HOME"],
    ):
        self.n_samples = n_samples
        self.base_model_name = base_model_name
        self.embed_size = embed_size
        self.epsilon = epsilon
        self.tau_prime = tau_prime
        self.seed = seed
        self.cache_dir = cache_dir

        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.base_model_name, cache_dir=self.cache_dir, device_map="auto"
        )
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.base_model_name, cache_dir=self.cache_dir
        )
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        for key, max_len in MODEL_MAX_LENGTH.items():
            if key in self.base_model_name:
                self.base_tokenizer.model_max_length = max_len
                break

    def get_samples(self, logits, labels):
        assert logits.shape[0] == 1
        assert labels.shape[0] == 1
        nsamples = self.n_samples
        lprobs = torch.log_softmax(logits, dim=-1)
        distrib = torch.distributions.categorical.Categorical(logits=lprobs)
        samples = distrib.sample([nsamples]).permute([1, 2, 0])
        return samples

    def get_likelihood(self, logits, labels):
        assert logits.shape[0] == 1
        assert labels.shape[0] == 1
        labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
        lprobs = torch.log_softmax(logits, dim=-1)
        log_likelihood = lprobs.gather(dim=-1, index=labels)
        return log_likelihood

    def get_lastde(self, log_likelihood):
        embed_size = self.embed_size
        epsilon = int(self.epsilon * log_likelihood.shape[1])
        tau_prime = self.tau_prime

        templl = log_likelihood.mean(dim=1)
        aggmde = fastMDE.get_tau_multiscale_DE(
            ori_data=log_likelihood,
            embed_size=embed_size,
            epsilon=epsilon,
            tau_prime=tau_prime,
        )
        lastde = templl / aggmde if aggmde is not None else None
        return lastde

    def get_sampling_discrepancy(self, logits_ref, logits_score, labels):
        assert logits_ref.shape[0] == 1
        assert logits_score.shape[0] == 1
        assert labels.shape[0] == 1
        if logits_ref.size(-1) != logits_score.size(-1):
            # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
            vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
            logits_ref = logits_ref[:, :, :vocab_size]
            logits_score = logits_score[:, :, :vocab_size]

        samples = self.get_samples(logits_ref, labels)
        log_likelihood_x = self.get_likelihood(logits_score, labels)
        log_likelihood_x_tilde = self.get_likelihood(logits_score, samples)

        # lastde
        lastde_x = self.get_lastde(log_likelihood_x)
        sampled_lastde = self.get_lastde(log_likelihood_x_tilde)

        # For valid sliding window
        if lastde_x is None or sampled_lastde is None:
            return 0.0

        miu_tilde = sampled_lastde.mean()
        sigma_tilde = sampled_lastde.std()
        discrepancy = (lastde_x - miu_tilde) / sigma_tilde

        return discrepancy.cpu().item()

    def detect(self, text: str) -> float:
        if len(text) == 0:
            return 0.0
        first_device = next(self.base_model.parameters()).device
        tokenized = self.base_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_token_type_ids=False,
        ).to(first_device)

        if tokenized.input_ids.shape[1] <= 1:
            return 0.0

        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = self.base_model(**tokenized).logits[:, :-1]
            logits_ref = logits_score

            original_crit = self.get_sampling_discrepancy(
                logits_ref, logits_score, labels
            )
        return original_crit


class LastdeDoublePlus:
    def __init__(self, base_model_name: str):
        self.base_model_name = base_model_name
        self.lastde_doubleplus_instance = LastdeDoublePlusModel(
            base_model_name=self.base_model_name
        )

    def inference(self, texts: list) -> list:
        predictions = []
        for text in tqdm.tqdm(texts):
            pred_score = self.lastde_doubleplus_instance.detect(text)
            predictions.append(pred_score)

        return {"lastde_doubleplus": predictions}
