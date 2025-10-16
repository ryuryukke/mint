import argparse
from tqdm import tqdm
from method import get_method
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from datasets import load_dataset
import os
import json
import numpy as np
import pandas as pd
from raid.utils import load_data
from src.config import (
    TASK2DOMAINS,
    ALL_METHODS,
    ALL_MIA_METHODS,
    ALL_DETECTION_METHODS,
    BASELINE_METHODS,
    MULTI_BASIC_MIA_METHODS,
    MODEL_NAME_MAP,
    SMALLER_MOEL_NAME_MAP,
)


def load_evaluation_data(
    task: str, domain: str, model_name: str, decon_setting: str
) -> tuple:
    if task == "mia":
        dataset = load_dataset(
            "iamgroot42/mimir",
            domain,
            split=decon_setting,
            trust_remote_code=True,
        )
        negatives, positives = dataset["nonmember"], dataset["member"]
        print(f"Size of nonmembers: {len(negatives)}, members: {len(positives)}")
    elif task == "detection":
        raid_path = "/data/RAID/raid_train.pkl"
        if os.path.exists(raid_path):
            train_df = pd.read_pickle(raid_path)
        else:
            train_df = load_data(split="train")
            with open(raid_path, "wb") as f:
                pd.to_pickle(train_df, f)
        negatives = train_df[
            (train_df["domain"] == domain)
            & (train_df["model"] == "human")
            & (train_df["attack"] == "none")
        ]
        positives = train_df[
            (train_df["domain"] == domain)
            & (train_df["model"] == model_name)
            & (train_df["repetition_penalty"] == "no")
            & (train_df["attack"] == "none")
            & (train_df["decoding"] == "sampling")
        ]
        print(f"Size of humans: {len(negatives)}, machines: {len(positives)}")
    return negatives, positives


def get_method_names(methods: str) -> list:
    if methods == "all":
        return ALL_METHODS
    elif methods == "all_mia":
        return ALL_MIA_METHODS
    elif methods == "all_detection":
        return ALL_DETECTION_METHODS
    else:
        return methods.split(" ")


def load_cache_path(task, domain, model_name, decon_setting):
    pred_dir = f"../results/{task}/predictions"
    score_dir = f"../results/{task}/scores"

    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(score_dir, exist_ok=True)

    common_name = f"{domain}_{model_name.replace('/', '_')}"
    if task == "mia":
        common_name += f"_{decon_setting}"

    pred_path = f"{pred_dir}/{common_name}.json"
    score_path = f"{score_dir}/{common_name}.json"

    return pred_path, score_path


def load_cached_results(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            results = json.load(f)
    else:
        results = dict()
    return results


def load_prefix(task: str, domain: str) -> dict:
    with open(
        f"../data/Pile/{'Prefix' if task == 'mia' else 'RAID'}/{domain}_prefix.json",
        "r",
    ) as f:
        prefix = json.load(f)
    return prefix


def get_roc_auc(real_preds: list, sample_preds: list) -> float:
    y_true = np.array([0] * len(real_preds) + [1] * len(sample_preds))
    y_scores = np.array(real_preds + sample_preds, dtype=float)

    mask = ~np.isnan(y_scores)
    y_true, y_scores = y_true[mask], y_scores[mask]

    # ROC-AUC算出
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return float(auc(fpr, tpr))


def update_results(
    method_name: str,
    all_predictions: dict,
    scores: dict,
    predictions_for_negatives: dict,
    predictions_for_positives: dict,
) -> tuple:
    if method_name == "baselines":
        methods = BASELINE_METHODS
    elif method_name == "multi_basic_mia":
        methods = MULTI_BASIC_MIA_METHODS
    else:
        methods = [method_name]

    for method in methods:
        predictions_for_negatives_ = predictions_for_negatives[method]
        predictions_for_positives_ = predictions_for_positives[method]
        all_predictions[method] = {
            "negatives": predictions_for_negatives_,
            "positives": predictions_for_positives_,
        }
        roc_auc, _ = get_roc_auc(predictions_for_negatives_, predictions_for_positives_)
        scores[method] = {"roc_auc": roc_auc}

    return all_predictions, scores


def save_results(results: dict, save_path: str) -> None:
    with open(save_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["mia", "detection"])
    parser.add_argument("--domain", type=str)
    parser.add_argument(
        "--methods",
        type=str,
        choices=[
            "all",
            "all_mia",
            "all_detection",
            "loss",
            "entropy",
            "rank",
            "logrank",
            "ref",
            "zlib",
            "min_k",
            "min_k_plus",
            "neighborhood",
            "recall",
            "dc_pdd",
            "detectgpt",
            "fastdetectgpt",
            "binoculars",
            "detectllm",
            "lastde_doubleplus",
            "multi_basic_mia",  # "loss", "entropy", "rank", "logrank", "ref", "zlib", "min_k", "min_k_plus"
            "baselines",  # "loss", "entropy", "rank", "logrank"
        ],
        help="You can also specify a space-separated list of methods.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=[
            "gpt2",
            "mpt-chat",
            "llama-chat",
            "chatgpt",
            "gpt4",
            "pythia-160m",
            "pythia-1.4b",
            "pythia-2.8b",
            "pythia-6.9b",
            "pythia-12b",
        ],
    )
    parser.add_argument("--decon_setting", type=str, default="13_02")

    args = parser.parse_args()

    assert args.domain in TASK2DOMAINS[args.task], (
        f"{args.domain} not in {TASK2DOMAINS[args.task]}, please specify the correct domain for {args.task} task."
    )

    print(
        f"Evaluate methods for {args.task} task, {args.domain} domain, {args.model_name} model"
    )

    task, domain, methods, model_name, decon_setting = (
        args.task,
        args.domain,
        args.methods,
        args.model_name,
        args.decon_setting,
    )

    negatives, positives = load_evaluation_data(task, domain, model_name, decon_setting)
    method_names = get_method_names(methods)
    pred_path, score_path = load_cache_path(task, domain, model_name, decon_setting)
    predictions, scores = (
        load_cached_results(pred_path),
        load_cached_results(score_path),
    )

    prefix = None
    model_name, ref_model_name = (
        MODEL_NAME_MAP.get(model_name, model_name),
        SMALLER_MOEL_NAME_MAP.get(model_name, model_name),
    )
    for method_name in tqdm(method_names):
        if method_name == "recall":
            prefix = load_prefix(domain)
        method = get_method(method_name, model_name, ref_model_name, prefix)

        predictions_for_negatives, predictions_for_positives = (
            method.inference(negatives),
            method.inference(positives),
        )

        predictions, scores = update_results(
            method_name,
            predictions,
            scores,
            predictions_for_negatives,
            predictions_for_positives,
        )

        # save prediction scores per methods
        save_results(predictions, pred_path)
        save_results(scores, score_path)
