from methods.loss.loss import Loss
from methods.likelihood.likelihood import Likelihood
from methods.entropy.entropy import Entropy
from methods.rank.rank import Rank
from methods.ref.ref import Reference
from methods.zlib.zlib import Zlib
from methods.min_k.min_k import MinK
from methods.min_k_plus.min_k_plus import MinKPlus
from methods.recall.recall import Recall
from methods.dc_pdd.dc_pdd import DCPDD
from methods.detectgpt.detectgpt import DetectGPT
from methods.fast_detectgpt.fast_detectgpt import FastDetectGPT
from methods.binoculars.binoculars import Binoculars
from methods.lastde_doubleplus.lastde_doubleplus import LastdeDoublePlus
from methods.multi_basic.multi_basic import MultiBasic
from methods.baselines.baselines import Baselines


class Method:
    """Shared interface for all methods."""

    def inference(self, texts: list) -> dict:
        """Takes in a list of texts and outputs a dict of scores from 0 to 1 with
        0 indicating likely non-member or human-written, and 1 indicating likely member or machine-generated with a method name as a key."""
        pass


def get_method(
    method_name: str, base_model_name: str, ref_model_name: str, prefix: dict
) -> Method:
    if method_name == "loss":
        return Loss(base_model_name)
    elif method_name == "likelihood":
        return Likelihood(base_model_name)
    elif method_name == "entropy":
        return Entropy(base_model_name)
    elif method_name == "rank":
        return Rank(base_model_name, log=False)
    elif method_name == "logrank":
        return Rank(base_model_name, log=True)
    elif method_name == "reference":
        return Reference(base_model_name, ref_model_name)
    elif method_name == "zlib":
        return Zlib(base_model_name)
    elif method_name == "neighborhood":
        return DetectGPT(base_model_name)
    elif method_name == "min_k":
        return MinK(base_model_name)
    elif method_name == "min_k_plus":
        return MinKPlus(base_model_name)
    elif method_name == "recall":
        return Recall(base_model_name, prefix)
    elif method_name == "dc_pdd":
        return DCPDD(base_model_name)
    elif method_name == "detectgpt":
        return DetectGPT(base_model_name)
    elif method_name == "fastdetectgpt":
        return FastDetectGPT(False, base_model_name, base_model_name)
    elif method_name == "binoculars":
        if base_model_name == "meta-llama/Llama-3.2-3B":
            return Binoculars("meta-llama/Llama-3.2-3B-instruct", base_model_name)
        elif base_model_name == "meta-llama/Llama-2-70b-chat-hf":
            return Binoculars("meta-llama/Llama-2-70b-hf", base_model_name)
        elif "pythia" in base_model_name:
            return Binoculars(base_model_name + "-deduped", base_model_name)
        elif "mpt" in base_model_name:
            return Binoculars("mosaicml/mpt-30b", base_model_name)
        elif "gpt2" in base_model_name:
            return Binoculars("lgaalves/gpt2-xl_lima", base_model_name)
    elif method_name == "detectllm":
        return FastDetectGPT(True, base_model_name, base_model_name)
    elif method_name == "lastde_doubleplus":
        return LastdeDoublePlus(base_model_name)
    elif (
        method_name == "multi_basic_mia"
    ):  # efficient single inference for multiple methods: loss, entropy, rank, logrank, reference, zlib, min_k, min_k_plus
        return MultiBasic(base_model_name, ref_model_name)
    elif (
        method_name == "baselines"
    ):  # efficient single inference for baselines: loss, entropy, rank, logrank
        return Baselines(base_model_name, ref_model_name)
    else:
        raise ValueError("Invalid method name")
