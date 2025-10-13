# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import glob
import json
import os

import torch

from .fast_detect_gpt import (
    get_log_sampling_discrepancy_analytic,
    get_sampling_discrepancy_analytic,
)
from .model import load_model, load_tokenizer
from src.config import MODEL_MAX_LENGTH


# estimate the probability according to the distribution of our test results on ChatGPT and GPT-4
class ProbEstimator:
    def __init__(self, ref_path: str, use_log_rank: bool):
        self.use_log_rank = use_log_rank
        self.ref_path = ref_path
        self.real_crits = []
        self.fake_crits = []
        for result_file in glob.glob(os.path.join(self.ref_path, "*.json")):
            with open(result_file, "r") as fin:
                res = json.load(fin)
                self.real_crits.extend(res["predictions"]["real"])
                self.fake_crits.extend(res["predictions"]["samples"])
        print(f"ProbEstimator: total {len(self.real_crits) * 2} samples.")


class FastDetectGPTModel:
    def __init__(
        self,
        scoring_model_name: str,
        reference_model_name: str,
        cache_dir: str,
        dataset: str,
        ref_path: str,
        use_log_rank: bool,
    ):
        self.dataset = dataset
        self.cache_dir = cache_dir
        self.scoring_model_name = scoring_model_name
        self.reference_model_name = reference_model_name
        self.ref_path = ref_path

        # load model
        self.scoring_tokenizer = load_tokenizer(
            self.scoring_model_name, self.dataset, self.cache_dir
        )
        for key, max_len in MODEL_MAX_LENGTH.items():
            if key in self.scoring_model_name:
                self.scoring_tokenizer.model_max_length = max_len
                break

        self.scoring_model = load_model(self.scoring_model_name, self.cache_dir)
        self.scoring_model.eval()
        if self.reference_model_name != self.scoring_model_name:
            self.reference_tokenizer = load_tokenizer(
                self.reference_model_name, self.dataset, self.cache_dir
            )
            for key, max_len in MODEL_MAX_LENGTH.items():
                if key in self.reference_model_name:
                    self.reference_tokenizer.model_max_length = max_len
                    break

            self.reference_model = load_model(self.reference_model_name, self.cache_dir)
            self.reference_model.eval()

        # evaluate criterion
        self.name = "sampling_discrepancy_analytic"
        if use_log_rank:
            self.criterion_fn = get_log_sampling_discrepancy_analytic
        else:
            self.criterion_fn = get_sampling_discrepancy_analytic

        self.prob_estimator = ProbEstimator(self.ref_path, use_log_rank)

    def run(self, text: str) -> float:
        first_device = next(self.scoring_model.parameters()).device
        tokenized = self.scoring_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_token_type_ids=False,
        ).to(first_device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = self.scoring_model(**tokenized).logits[:, :-1]
            if self.reference_model_name == self.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = self.reference_tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    return_token_type_ids=False,
                ).to(first_device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), (
                    "Tokenizer is mismatch."
                )
                logits_ref = self.reference_model(**tokenized).logits[:, :-1]
            # save VRAM memory
            crit = self.criterion_fn(
                logits_ref.detach().to("cpu", dtype=torch.float32),
                logits_score.detach().to("cpu", dtype=torch.float32),
                labels.detach().to("cpu"),
            )

        return crit
