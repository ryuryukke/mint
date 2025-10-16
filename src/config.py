TASK2DOMAINS = {
    "mia": [
        "wikipedia_(en)",
        "github",
        "pile_cc",
        "pubmed_central",
        "arxiv",
        "dm_mathematics",
    ],
    "detection": [
        "abstracts",
        "books",
        "news",
        "poetry",
        "recipes",
        "reddit",
        "reviews",
        "wiki",
    ],
}

# Set model_max_length for models without a defined default value.
MODEL_MAX_LENGTH = {
    "pythia": 2048,
    "gpt2": 1024,
    "mpt": 512,
    "Llama-2": 512,
    "Llama-3": 4096,
}

# all methods with efficient single inference: multi_basic_mia
ALL_METHODS = [
    "multi_basic_mia",
    "neighborhood",
    "recall",
    "dc_pdd",
    "detectgpt",
    "fastdetectgpt",
    "binoculars",
    "detectllm",
    "lastde_doubleplus",
]

# all mia methods with efficient single inference: multi_basic_mia
ALL_MIA_METHODS = ["multi_basic_mia", "neighborhood", "recall", "dc_pdd"]

# all detection methods with efficient single inference
ALL_DETECTION_METHODS = [
    "baselines",
    "detectgpt",
    "fastdetectgpt",
    "binoculars",
    "detectllm",
    "lastde_doubleplus",
]

# baseline methods
BASELINE_METHODS = ["loss", "entropy", "rank", "logrank"]

# mia methods with efficient single inference
MULTI_BASIC_MIA_METHODS = [
    "loss",
    "entropy",
    "rank",
    "logrank",
    "ref",
    "zlib",
    "min_k",
    "min_k_plus",
]

# mapping from model name to actual model path in Hugging Face
MODEL_NAME_MAP = {
    "gpt2": "openai-community/gpt2-xl",
    "mpt-chat": "mosaicml/mpt-30b-chat",
    "llama-chat": "meta-llama/Llama-2-70b-chat-hf",
    "chatgpt": "EleutherAI/pythia-160m",  # "meta-llama/Llama-3.2-3B"
    "gpt4": "EleutherAI/pythia-160m",  # "meta-llama/Llama-3.2-3B"
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "pythia-2.8b": "EleutherAI/pythia-2.8b",
    "pythia-6.9b": "EleutherAI/pythia-6.9b",
    "pythia-12b": "EleutherAI/pythia-12b",
}

# mapping from model name to smaller model path in Hugging Face for reference models
SMALLER_MOEL_NAME_MAP = {
    "gpt2": "openai-community/gpt2",
    "mpt-chat": "mosaicml/mpt-7b-chat",
    "llama-chat": "meta-llama/Llama-2-7b-chat-hf",
    "chatgpt": "EleutherAI/pythia-70m",  # "meta-llama/Llama-3.2-1B"
    "gpt4": "EleutherAI/pythia-70m",  # "meta-llama/Llama-3.2-1B"
    "pythia-160m": "EleutherAI/pythia-70m",
    "pythia-1.4b": "EleutherAI/pythia-70m",
    "pythia-2.8b": "EleutherAI/pythia-70m",
    "pythia-6.9b": "EleutherAI/pythia-70m",
    "pythia-12b": "EleutherAI/pythia-70m",
}
