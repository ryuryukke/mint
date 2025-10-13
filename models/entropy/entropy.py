import os
import torch
import tqdm
import transformers
import torch.nn.functional as F
from src.config import MODEL_MAX_LENGTH


class EntropyModel:
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
            if key in self.base_model_name:
                self.base_tokenizer.model_max_length = max_len
                break

    def detect(self, text: str) -> float:
        if len(text) == 0:
            return 0.0
        with torch.no_grad():
            first_device = next(self.base_model.parameters()).device
            tokenized = self.base_tokenizer(
                text, truncation=True, return_tensors="pt"
            ).to(first_device)
            logits = self.base_model(**tokenized).logits[:, :-1]
            neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
            return -neg_entropy.sum(-1).mean().item()


class Entropy:
    def __init__(self, base_model_name: str):
        self.base_model_name = base_model_name
        self.entropy_instance = EntropyModel(
            self.base_model_name, cache_dir=os.environ["HF_HOME"]
        )

    def inference(self, texts: list) -> list:
        predictions = []
        for text in tqdm.tqdm(texts):
            pred_score = -self.entropy_instance.detect(text)
            predictions.append(pred_score)

        return predictions
