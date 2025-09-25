from .models.binoculars.binoculars import Binoculars
from .models.detectgpt.detectgpt import DetectGPT
from .models.fast_detectgpt.fast_detectgpt import FastDetectGPT
from .models.likelihood.likelihood import Likelihood
from .models.rank.rank import Rank
from .models.entropy.entropy import Entropy
from .models.loss.loss import Loss
from .models.ref.ref import Reference
from .models.zlib.zlib import Zlib
from .models.min_k.min_k import MinK
from .models.min_k_plus.min_k_plus import MinKPlus
from .models.logit_based.logit_based import LogitBased
from .models.recall.recall import Recall
from .models.dc_pdd.dc_pdd import DCPDD
from .models.lastde_doubleplus.lastde_doubleplus import LastdeDoublePlus

class Detector:
    """Shared interface for all detectors"""

    def inference(self, texts: list) -> list:
        """Takes in a list of texts and outputs a list of scores from 0 to 1 with
        0 indicating likely human-written, and 1 indicating likely machine-generated."""
        pass

binoculars_llama_model_pair = {
    "huggyllama/llama-13b": "lmsys/vicuna-13b-v1.3",
    "huggyllama/llama-30b": "lmsys/vicuna-33b-v1.3",
    "huggyllama/llama-65b": "lmsys/vicuna-33b-v1.3",
    "meta-llama/Llama-3.2-3B": "meta-llama/Llama-3.2-3B-instruct",
    "meta-llama/Llama-2-70b-chat-hf": "meta-llama/Llama-2-70b-hf"}


def get_detector(
    detector_name: str, base_model_name: str, ref_model_name: str, prefix: dict
) -> Detector:
    if detector_name == "detectgpt":
        return DetectGPT(base_model_name)
    elif detector_name == "fastdetectgpt":
            return FastDetectGPT(False, base_model_name, base_model_name)
    elif detector_name == "fastdetectllm":
            return FastDetectGPT(True, base_model_name, base_model_name)
    elif detector_name == "binoculars":
        if base_model_name in binoculars_llama_model_pair:
            return Binoculars(binoculars_llama_model_pair[base_model_name], base_model_name)
        elif "mistral" in base_model_name:
            return Binoculars("mistralai/Mistral-7B-v0.1", base_model_name)
        elif 'pythia' in base_model_name:
            return Binoculars(base_model_name + "-deduped", base_model_name)
        elif 'neox' in base_model_name:
            return Binoculars('togethercomputer/GPT-NeoXT-Chat-Base-20B', base_model_name)
        elif "mpt" in base_model_name:
            return Binoculars("mosaicml/mpt-30b", base_model_name)
        elif "gpt2" in base_model_name:
            return Binoculars("lgaalves/gpt2-xl_lima", base_model_name)
    elif detector_name == "logit_based":
        return LogitBased(base_model_name, ref_model_name)
    elif detector_name == "recall":
        return Recall(base_model_name, prefix)
    elif detector_name == "dc_pdd":
        return DCPDD(base_model_name)
    elif detector_name == "lastde_doubleplus":
        return LastdeDoublePlus(base_model_name)
    else:
        raise ValueError("Invalid detector name")
