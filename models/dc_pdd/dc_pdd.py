"""
Adapted from the original implementation of DC-PDD: https://github.com/zhang-wei-chao/DC-PDD
Some modifications added.
"""

import os
import numpy as np
import torch
from tqdm import tqdm
import transformers
import pickle
from src.config import MODEL_MAX_LENGTH


class DCPDDModel:
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

        # using the defaults of the original implementation
        self.file_num = 15
        self.max_tok = 1024
        self.a = 1e-10

    def cal_ppl(self, text: str) -> tuple:
        first_device = next(self.base_model.parameters()).device
        input_ids = self.base_tokenizer.encode(text, truncation=True)
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        input_ids = input_ids.to(first_device)
        with torch.no_grad():
            output = self.base_model(input_ids, labels=input_ids)
        logit = output[1]
        # Apply softmax to the logits to get probabilities
        prob = torch.nn.functional.log_softmax(logit, dim=-1)[0][:-1]
        input_ids = input_ids[0][1:]
        probs = prob[torch.arange(len(prob)).to(first_device), input_ids].tolist()
        input_ids = input_ids.cpu().numpy()
        return probs, input_ids

    def detect(self, text: str, freq_dist: dict) -> float:
        if len(text) == 0:
            return 0.0
        # compute the probability distribution
        probs, input_ids = self.cal_ppl(text)
        # compute the prediction score by calibrating the probability with the token frequency distribution
        probs = np.exp(probs)
        indexes = []
        current_ids = []
        for i, input_id in enumerate(input_ids):
            if input_id not in current_ids:
                indexes.append(i)
                current_ids.append(input_id)

        x_pro = probs[indexes]
        x_fre = np.array(freq_dist)[input_ids[indexes].tolist()]
        # To avoid zero-division:
        epsilon = 1e-10
        x_fre = np.where(x_fre == 0, epsilon, x_fre)
        ce = x_pro * np.log(1 / x_fre)
        ce[ce > self.a] = self.a
        return -np.mean(ce)


class DCPDD:
    def __init__(self, base_model_name: str):
        self.base_model_name = base_model_name
        self.dc_pdd_instance = DCPDDModel(
            self.base_model_name, cache_dir=os.environ["HF_HOME"]
        )

    def load_freq_dist(self):
        for model_type in ["pythia", "gpt2", "mpt", "mistral", "Llama-2", "Llama-3"]:
            if model_type in self.base_model_name:
                dist_path = f"../data/C4/freq_dist_{model_type}.pkl"
                if not os.path.exists(dist_path):
                    raise ValueError(
                        "Please build the token frequency distribution first via /data/C4/build_token_dist.sh"
                    )
                with open(dist_path, "rb") as f:
                    freq_dist = pickle.load(f)
                return freq_dist

    def inference(self, texts: list) -> list:
        freq_dist = self.load_freq_dist()
        predictions = []
        for text in tqdm(texts):
            pred_score = self.dc_pdd_instance.detect(text, freq_dist)
            predictions.append(pred_score)

        return predictions
