import os
import torch
import tqdm
import transformers
import zlib
from src.config import MODEL_MAX_LENGTH


class ZlibModel:
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
        with torch.no_grad():
            first_device = next(self.base_model.parameters()).device
            tokenized = self.base_tokenizer(
                text, truncation=True, return_tensors="pt"
            ).to(first_device)
            labels = tokenized.input_ids
            zlib_entropy = len(zlib.compress(bytes(text, "utf-8")))
            return (
                -self.base_model(**tokenized, labels=labels).loss.item() / zlib_entropy
            )


class Zlib:
    def __init__(self, base_model_name: str):
        self.base_model_name = base_model_name
        self.zlib_instance = ZlibModel(
            self.base_model_name, cache_dir=os.environ["HF_HOME"]
        )

    def inference(self, texts: list) -> list:
        predictions = []
        for text in tqdm.tqdm(texts):
            pred_score = self.zlib_instance.detect(text)
            predictions.append(pred_score)

        return predictions
