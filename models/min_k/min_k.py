import os
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import transformers
from src.config import MODEL_MAX_LENGTH


class MinKModel:
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

    def detect(self, text: str) -> float:
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

        return {"min_k": predictions}
