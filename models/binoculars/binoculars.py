import os
import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils.metrics import entropy, perplexity
from .utils.utils import assert_tokenizer_consistency

torch.set_grad_enabled(False)


class Binoculars(object):
    def __init__(
        self,
        observer_name_or_path: str,
        performer_name_or_path: str,
        use_bfloat16: bool = True,
        max_token_observed: int = 512,
    ) -> None:
        if "mpt" not in observer_name_or_path:
            assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)

        self.observer_model = AutoModelForCausalLM.from_pretrained(
            observer_name_or_path,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=os.environ["HF_HOME"],
            torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
        )
        self.performer_model = AutoModelForCausalLM.from_pretrained(
            performer_name_or_path,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=os.environ["HF_HOME"],
            torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
        )

        self.observer_model.eval()
        self.performer_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_token_observed = max_token_observed

    def _tokenize(self, batch: list) -> transformers.BatchEncoding:
        batch_size = len(batch)
        first_device = next(self.observer_model.parameters()).device
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False,
        ).to(first_device)
        return encodings

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> torch.Tensor:
        first_device = next(self.observer_model.parameters()).device
        second_device = next(self.performer_model.parameters()).device
        observer_logits = self.observer_model(**encodings.to(first_device)).logits
        performer_logits = self.performer_model(**encodings.to(second_device)).logits
        if first_device != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    def compute_score(self, input_text: str) -> float:
        batch = [input_text] if isinstance(input_text, str) else input_text
        first_device = next(self.observer_model.parameters()).device
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(
            observer_logits.to(first_device),
            performer_logits.to(first_device),
            encodings.to(first_device),
            self.tokenizer.pad_token_id,
        )
        binoculars_scores = ppl / x_ppl
        binoculars_scores = binoculars_scores.tolist()
        return (
            binoculars_scores[0] if isinstance(input_text, str) else binoculars_scores
        )

    def inference(self, texts: list) -> list:
        predictions = []
        for text in tqdm(texts):
            predictions.append(1 - self.compute_score(text))
        return predictions
