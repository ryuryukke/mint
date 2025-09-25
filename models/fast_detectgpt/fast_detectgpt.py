import os

from tqdm import tqdm

from .fastdetectgpt.local_infer import FastDetectGPTModel


class FastDetectGPT:
    def __init__(self, use_log_rank, base_model_name, ref_model_name):
        self.base_model_name = base_model_name
        self.ref_model_name = ref_model_name
        self.use_log_rank = use_log_rank
        parent_dir = os.path.dirname(__file__)
        ref_path = os.path.join(parent_dir, "fastdetectgpt/local_infer_ref")
        self.fast_detect_gpt_instance = FastDetectGPTModel(
            scoring_model_name=self.base_model_name,
            reference_model_name=self.ref_model_name,
            cache_dir=os.environ["HF_HOME"],
            dataset="xsum",
            ref_path=ref_path,
            use_log_rank=use_log_rank,
        )

    def inference(self, texts: list) -> list:
        predictions = []
        for text in tqdm(texts):
            prob = (
                -self.fast_detect_gpt_instance.run(text)
                if self.use_log_rank
                else self.fast_detect_gpt_instance.run(text)
            )
            predictions.append(prob)
        return predictions

    def interactive(self):
        self.fast_detect_gpt_instance.run_interactive()
