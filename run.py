import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import roc_curve, auc

from method import get_method
from raid.utils import load_data
from src.config import (
    TASK2DOMAINS,
    ALL_METHODS,
    ALL_MIA_METHODS,
    ALL_DETECTION_METHODS,
    BASELINE_METHODS,
    MULTI_BASIC_MIA_METHODS,
    MODEL_NAME_MAP,
    SMALLER_MODEL_NAME_MAP,
)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


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
        raid_path = DATA_DIR / "RAID" / "raid_train.pkl"
        if raid_path.exists():
            train_df = pd.read_pickle(raid_path)
        else:
            train_df = load_data(split="train")
            pd.to_pickle(train_df, raid_path)

        humans = train_df[
            (train_df["domain"] == domain)
            & (train_df["model"] == "human")
            & (train_df["attack"] == "none")
        ]
        machines = train_df[
            (train_df["domain"] == domain)
            & (train_df["model"] == model_name)
            & (train_df["repetition_penalty"] == "no")
            & (train_df["attack"] == "none")
            & (train_df["decoding"] == "sampling")
        ]
        negatives, positives = (
            humans["generation"].tolist(),
            machines["generation"].tolist(),
        )
        print(f"Size of humans: {len(negatives)}, machines: {len(positives)}")
    else:
        raise ValueError(f"Unknown task: {task}")
    return negatives, positives


def get_method_names(methods: str) -> list:
    return {
        "all": ALL_METHODS,
        "all_mia": ALL_MIA_METHODS,
        "all_detection": ALL_DETECTION_METHODS,
    }.get(methods, methods.split())


def load_cache_path(task, domain, model_name, decon_setting) -> tuple[Path, Path]:
    base = RESULTS_DIR / task
    pred_dir = base / "predictions"
    score_dir = base / "scores"
    pred_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    common = f"{domain}_{model_name.replace('/', '_')}"
    if task == "mia":
        common += f"_{decon_setting}"
    return pred_dir / f"{common}.json", score_dir / f"{common}.json"


def load_cached_results(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def load_prefix(task: str, domain: str, generator: str) -> dict:
    prefix_root = DATA_DIR / ("Pile" if task == "mia" else "RAID") / "Prefix"
    name = f"{domain}{'_prefix' if task == 'mia' else f'_{generator}_prefix'}.json"
    path = prefix_root / name
    if not path.exists():
        raise FileNotFoundError(
            f"Prefix file not found: {path}\n"
            f"Please prepare it by executing {(DATA_DIR / ('Pile' if task == 'mia' else 'RAID') / 'collect_prefix.sh')}"
        )
    return json.loads(path.read_text())


def get_roc_auc(real_preds: list, sample_preds: list) -> float:
    y_true = np.array([0] * len(real_preds) + [1] * len(sample_preds))
    y_scores = np.array(real_preds + sample_preds, dtype=float)

    mask = ~np.isnan(y_scores)
    y_true, y_scores = y_true[mask], y_scores[mask]

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return float(auc(fpr, tpr))


def update_results(
    method_name: str,
    all_predictions: dict,
    scores: dict,
    predictions_for_negatives: dict,
    predictions_for_positives: dict,
) -> tuple:
    methods = {
        "baselines": BASELINE_METHODS,
        "multi_basic_mia": MULTI_BASIC_MIA_METHODS,
    }.get(method_name, [method_name])

    for method in methods:
        predictions_for_negatives_ = predictions_for_negatives[method]
        predictions_for_positives_ = predictions_for_positives[method]
        all_predictions[method] = {
            "negatives": predictions_for_negatives_,
            "positives": predictions_for_positives_,
        }
        roc_auc = get_roc_auc(predictions_for_negatives_, predictions_for_positives_)
        scores[method] = {"roc_auc": roc_auc}

    return all_predictions, scores


def save_results(results: dict, save_path: Path) -> None:
    tmp = save_path.with_suffix(save_path.suffix + ".tmp")
    tmp.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    tmp.replace(save_path)


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
        help="Specify one or more methods (space-separated).",
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
    parser.add_argument("--decon_setting", type=str, default="ngram_13_0.2")

    args = parser.parse_args()

    if args.domain not in TASK2DOMAINS[args.task]:
        raise ValueError(
            f"Invalid domain '{args.domain}' for task '{args.task}'. "
            f"Expected one of: {', '.join(TASK2DOMAINS[args.task])}"
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
    if model_name not in MODEL_NAME_MAP:
        raise ValueError(f"Model {model_name} not supported.")

    real_model_name, real_ref_model_name = (
        MODEL_NAME_MAP[model_name],
        SMALLER_MODEL_NAME_MAP[model_name],
    )
    for method_name in tqdm(method_names):
        if method_name == "recall":
            prefix = load_prefix(task, domain, model_name)
        method = get_method(method_name, real_model_name, real_ref_model_name, prefix)

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

        save_results(predictions, pred_path)
        save_results(scores, score_path)
