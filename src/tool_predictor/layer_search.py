import os

seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random
import json
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


set_seed(seed)

import warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModel

from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

import xgboost as xgb
import pandas as pd


# -----------------------
# Utilities (copied from fix_seed.py)
# -----------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_csv_rows(path: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def _find_score_column(df: pd.DataFrame) -> str:
    candidate_cols = ["score"]
    for col in candidate_cols:
        if col in df.columns:
            return col
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError(
            "Could not find a numeric score column in the CSV. "
            "Expected one of: score, entity_hallucination_score, hallucination_score, summary_score, pred_score."
        )
    return numeric_cols[0]


def read_labels_from_path(
    predictor_id,
    path: str,
    score_threshold_1: float = 0.9,
    score_threshold_2: float = 0.1,
) -> Tuple[List[int], List[int], Optional[List[float]]]:
    ext = os.path.splitext(path)[1].lower()

    labels: List[int] = []
    valid_indices: List[int] = []

    if predictor_id == 1:
        if ext == ".csv":
            df = pd.read_csv(path)
            score_col = _find_score_column(df)
            scores = pd.to_numeric(df[score_col], errors="coerce")

            valid_scores: List[float] = []
            for i, score in enumerate(scores.tolist()):
                if pd.isna(score):
                    print("score column is empty")
                    continue
                label = 1 if float(score) > score_threshold_1 else 0
                labels.append(label)
                print(f"The Score is {score}, Labeled as {label}")
                valid_indices.append(i)
                valid_scores.append(float(score))
            return labels, valid_indices, valid_scores

    elif predictor_id == 2 or predictor_id == 3 or predictor_id == 4:
        if ext == ".csv":
            if "no_search" not in path:
                raise ValueError(
                    f"predictor_id=2 expects a no-search csv path, got: {path}"
                )

            force_search_path = path.replace("no_search", "force_search")

            if force_search_path == path:
                raise ValueError(
                    f"Failed to derive force_search path from input path: {path}"
                )

            if not os.path.exists(force_search_path):
                raise FileNotFoundError(
                    f"Corresponding force_search csv not found: {force_search_path}"
                )

            df_no_search = pd.read_csv(path)
            df_force_search = pd.read_csv(force_search_path)

            score_col_no_search = _find_score_column(df_no_search)
            score_col_force_search = _find_score_column(df_force_search)

            scores_no_search = pd.to_numeric(
                df_no_search[score_col_no_search], errors="coerce"
            )
            scores_force_search = pd.to_numeric(
                df_force_search[score_col_force_search], errors="coerce"
            )

            if len(scores_no_search) != len(scores_force_search):
                raise ValueError(
                    "no-search and force-search csv files must have the same number of rows: "
                    f"{len(scores_no_search)} != {len(scores_force_search)}"
                )

            def _bucketize(score: float) -> int:
                return score

            for i, (score_no, score_force) in enumerate(
                zip(scores_no_search.tolist(), scores_force_search.tolist())
            ):
                if pd.isna(score_no):
                    print(f"Row {i}: no_search score is NaN, padding as 0")
                    score_no = 0.0
                if pd.isna(score_force):
                    print(f"Row {i}: force_search score is NaN, padding as 0")
                    score_force = 0.0

                bucket_no = _bucketize(float(score_no))
                bucket_force = _bucketize(float(score_force))

                if bucket_force > bucket_no:
                    label = 1
                elif bucket_force == bucket_no:
                    label = 0
                else:
                    label = 0

                labels.append(label)
                valid_indices.append(i)
                print(
                    f"The no search score is {score_no}\n"
                    f"The force search score is {score_force}\n"
                    f"The label is: {label}\n"
                    f"=========================\n"
                )

            return labels, valid_indices, None

    return labels, valid_indices, None


def safe_get_prompt(item: Dict[str, Any]) -> Optional[str]:
    if "prompt" in item and isinstance(item["prompt"], str):
        return item["prompt"]
    if "query" in item and isinstance(item["query"], str):
        return item["query"]
    return None


def detect_setting_from_filename(results_path: str) -> str:
    fn = os.path.basename(results_path)
    if "no_search" in fn:
        return "no_search"
    if "with_search" in fn:
        return "with_search"
    return "unknown"


def sanitize_model_name(model_path_or_name: str) -> str:
    base = os.path.basename(model_path_or_name.rstrip("/"))
    if base:
        return base
    return model_path_or_name.replace("/", "_").replace(":", "_")


def build_input_text(
    item: Dict[str, Any],
    setting: str,
    predictor_id,
    is_training: bool = False,
) -> Optional[str]:
    prompt = safe_get_prompt(item)
    if prompt is None and predictor_id in (1, 2, 3):
        return None

    if predictor_id == 1 or predictor_id == 3:
        return prompt

    if predictor_id == 2:
        if setting == "with_search":
            tipi = item.get("tool_in_prompt_info", "") or ""
            if not isinstance(tipi, str):
                tipi = str(tipi)
            return tipi
        else:
            return prompt

    if predictor_id == 4:
        return item.get("final_prompt", "") or ""

    raise ValueError(f"Unknown predictor_id: {predictor_id}")


def build_dataset_by_original_index(
    rows: List[Dict[str, Any]],
    valid_indices: List[int],
    labels: List[int],
    setting: str,
    predictor_id: int,
) -> Tuple[Dict[int, str], Dict[int, int]]:
    texts_by_idx: Dict[int, str] = {}
    labels_by_idx: Dict[int, int] = {}

    for idx, y_val in zip(valid_indices, labels):
        r = rows[idx]
        text = build_input_text(
            r,
            setting=setting,
            predictor_id=predictor_id,
            is_training=True,
        )
        if text is None:
            continue
        texts_by_idx[idx] = text
        labels_by_idx[idx] = int(y_val)

    return texts_by_idx, labels_by_idx


def save_predictions_csv(
    out_dir: str,
    clf_name: str,
    y_test,
    y_pred,
    y_pred_probs=None,
    task: str = "classification",
    texts_test: Optional[List[str]] = None,
    layer_idx: Optional[int] = None,
    test_original_indices: Optional[List[int]] = None,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    records = []
    for i, (yt, yp) in enumerate(zip(y_test, y_pred)):
        row: Dict[str, Any] = {
            "sample_index": i,
            "original_row_index": (
                test_original_indices[i]
                if test_original_indices is not None and i < len(test_original_indices)
                else ""
            ),
            "raw_input": texts_test[i] if texts_test is not None and i < len(texts_test) else "",
            "ground_truth_label": yt,
            "prediction": yp,
        }
        if y_pred_probs is not None and task == "classification":
            for k, prob in enumerate(y_pred_probs[i]):
                row[f"prob_class_{k}"] = round(float(prob), 6)
        records.append(row)
    df = pd.DataFrame(records)
    layer_tag = f"_layer{layer_idx}" if layer_idx is not None else ""
    csv_path = os.path.join(out_dir, f"predictions_{clf_name}{layer_tag}.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


# -----------------------
# Classifiers
# -----------------------

def _confidence_scores(y_test, y_pred, y_pred_probs) -> Dict:
    classes = sorted(set(int(l) for l in y_test) | set(int(l) for l in y_pred))
    all_probs: Dict[int, Dict[int, List]] = {
        c: {d: [] for d in classes} for c in classes
    }
    for true_label, pred_label, probs in zip(y_test, y_pred, y_pred_probs):
        tl, pl = int(true_label), int(pred_label)
        if tl in all_probs and pl in all_probs[tl]:
            all_probs[tl][pl].append(float(max(probs)))
    confidence_scores: Dict = {}
    for tl in classes:
        confidence_scores[tl] = {}
        for pl in classes:
            p = all_probs[tl][pl]
            confidence_scores[tl][pl] = (
                round(float(np.mean(p)), 4) if p else 0.0,
                round(float(np.std(p)), 4) if p else 0.0,
            )
    return confidence_scores


def _base_results(y_test, y_pred, y_pred_probs, search) -> Dict[str, Any]:
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    num_classes = len(np.unique(y_test))
    if num_classes == 2:
        roc_auc = round(float(roc_auc_score(y_test, y_pred_probs[:, 1])), 4)
    else:
        roc_auc = round(float(roc_auc_score(y_test, y_pred_probs, multi_class="ovr", average="macro")), 4)
    conf_scores = _confidence_scores(y_test, y_pred, y_pred_probs)
    return {
        "best_params": search.best_params_,
        "report": report,
        "confusion_matrix": conf_matrix,
        "roc_auc": roc_auc,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred)), 4),
        "confidence_scores": conf_scores,
        "y_pred": y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred),
        "y_pred_probs": y_pred_probs.tolist() if hasattr(y_pred_probs, "tolist") else list(y_pred_probs),
    }


def logistic_regression_clf(X_train, y_train, X_test, y_test, seed: int) -> Dict[str, Any]:
    pipeline = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=-1, random_state=seed)),
    ])
    grid = {
        "clf__C": [0.01, 0.1, 1.0],
        "clf__penalty": ["l2"],
        "clf__class_weight": ["balanced"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = GridSearchCV(pipeline, grid, cv=cv, n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)
    y_pred = search.predict(X_test)
    y_pred_probs = search.predict_proba(X_test)
    return _base_results(y_test, y_pred, y_pred_probs, search)


def mlp_clf(X_train, y_train, X_test, y_test, seed: int) -> Dict[str, Any]:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            max_iter=100,
            early_stopping=True,
            n_iter_no_change=5,
            random_state=seed,
        )),
    ])
    grid = {
        "clf__hidden_layer_sizes": [(), (128,), (256,), (128, 64), (1024, 64)],
        "clf__learning_rate_init": [1e-3, 1e-4],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = GridSearchCV(pipeline, grid, cv=cv, n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)
    y_pred = search.predict(X_test)
    y_pred_probs = search.predict_proba(X_test)
    return _base_results(y_test, y_pred, y_pred_probs, search)


def xgboost_clf(X_train, y_train, X_test, y_test, seed: int) -> Dict[str, Any]:
    unique_classes = np.unique(y_train)
    num_classes = len(unique_classes)
    is_binary = num_classes == 2

    label_to_idx = {lbl: idx for idx, lbl in enumerate(sorted(unique_classes))}
    idx_to_label = {idx: lbl for lbl, idx in label_to_idx.items()}
    y_train_enc = np.array([label_to_idx[l] for l in y_train], dtype=np.int64)

    xgb_params = dict(
        random_state=seed,
        eval_metric="mlogloss" if not is_binary else "logloss",
        verbosity=0,
    )
    if not is_binary:
        xgb_params["num_class"] = num_classes
        xgb_params["objective"] = "multi:softprob"

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", xgb.XGBClassifier(**xgb_params)),
    ])
    grid = {
        "clf__max_depth": [2, 6, 10],
        "clf__n_estimators": [10, 100],
    }
    if is_binary:
        neg = int((y_train_enc == 0).sum())
        pos = int((y_train_enc == 1).sum())
        if pos > 0:
            pipeline.set_params(clf__scale_pos_weight=neg / pos)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = GridSearchCV(pipeline, grid, cv=cv, n_jobs=-1, verbose=0)
    search.fit(X_train, y_train_enc)

    y_pred_enc = search.predict(X_test)
    y_pred_probs = search.predict_proba(X_test)
    y_pred_orig = np.array([idx_to_label[i] for i in y_pred_enc])
    return _base_results(y_test, y_pred_orig, y_pred_probs, search)


def train_and_report(
    X_all: np.ndarray,
    y_all,
    n_splits: int = 5,
    clf_names: List[str] = ("LogisticRegression", "MLP", "XGBoost"),
    out_dir: Optional[str] = None,
    texts_all: Optional[List[str]] = None,
    all_original_indices: Optional[List[int]] = None,
    seed: int = 42,
    layer_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run n_splits-fold CV for each classifier on X_all embeddings.
    Returns summary dict keyed by classifier name.
    """
    dispatcher = {
        "LogisticRegression": lambda Xtr, ytr, Xte, yte: logistic_regression_clf(Xtr, ytr, Xte, yte, seed),
        "MLP": lambda Xtr, ytr, Xte, yte: mlp_clf(Xtr, ytr, Xte, yte, seed),
        "XGBoost": lambda Xtr, ytr, Xte, yte: xgboost_clf(Xtr, ytr, Xte, yte, seed),
    }

    y_all = np.asarray(y_all)
    n_samples = len(y_all)
    n_classes = len(np.unique(y_all))

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(kf.split(np.zeros(n_samples), y_all))

    summary: Dict[str, Any] = {}

    for clf_name in clf_names:
        if clf_name not in dispatcher:
            raise ValueError(f"Unknown classifier: {clf_name}. Choose from {list(dispatcher)}")

        layer_tag = f" [layer {layer_idx}]" if layer_idx is not None else ""
        print(f"\n  ── {clf_name}{layer_tag}: {n_splits}-fold CV ──")

        oof_y_true = np.empty(n_samples, dtype=y_all.dtype)
        oof_y_pred = np.empty(n_samples, dtype=np.int64)
        oof_y_pred_probs = np.empty((n_samples, n_classes), dtype=np.float64)
        fold_rocs: List[float] = []

        for fold_i, (train_idx, test_idx) in enumerate(folds):
            res = dispatcher[clf_name](
                X_all[train_idx], y_all[train_idx],
                X_all[test_idx],  y_all[test_idx],
            )
            oof_y_true[test_idx] = y_all[test_idx]
            oof_y_pred[test_idx] = np.asarray(res["y_pred"])
            oof_y_pred_probs[test_idx] = np.asarray(res["y_pred_probs"])
            fold_rocs.append(res["roc_auc"])
            print(f"    fold {fold_i + 1}/{n_splits} | AUROC={res['roc_auc']:.4f} "
                  f"Acc={res['accuracy']:.4f} BalAcc={res['balanced_accuracy']:.4f}")

        mean_roc = float(np.mean(fold_rocs))
        if n_classes == 2:
            oof_roc_auc = round(float(roc_auc_score(oof_y_true, oof_y_pred_probs[:, 1])), 4)
        else:
            oof_roc_auc = round(float(roc_auc_score(oof_y_true, oof_y_pred_probs, multi_class="ovr", average="macro")), 4)

        result = {
            "mean_fold_roc_auc": round(mean_roc, 4),
            "fold_roc_aucs": [round(r, 4) for r in fold_rocs],
            "oof_roc_auc": oof_roc_auc,
            "accuracy": round(float(accuracy_score(oof_y_true, oof_y_pred)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(oof_y_true, oof_y_pred)), 4),
            "report": classification_report(oof_y_true, oof_y_pred, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(oof_y_true, oof_y_pred).tolist(),
            "y_pred": oof_y_pred.tolist(),
            "y_pred_probs": oof_y_pred_probs.tolist(),
            "y_true": oof_y_true.tolist(),
        }

        print(
            f"\n  ★ {clf_name}{layer_tag} | mean_fold_AUROC={result['mean_fold_roc_auc']:.4f} "
            f"OOF_AUROC={result['oof_roc_auc']:.4f} "
            f"Acc={result['accuracy']:.4f} BalAcc={result['balanced_accuracy']:.4f}"
        )

        if out_dir is not None:
            csv_path = save_predictions_csv(
                out_dir,
                clf_name,
                y_test=result["y_true"],
                y_pred=result["y_pred"],
                y_pred_probs=np.array(result["y_pred_probs"]),
                task="classification",
                texts_test=texts_all,
                layer_idx=layer_idx,
                test_original_indices=all_original_indices,
            )
            print(f"    OOF predictions ({n_samples} samples) saved to: {csv_path}")

        summary[clf_name] = {
            "result": {k: v for k, v in result.items() if k not in ("y_pred", "y_pred_probs", "y_true")},
        }

    return summary


# -----------------------
# All-layers embedder
# -----------------------

@dataclass
class EmbedConfig:
    model_name_or_path: str
    device: str
    batch_size: int = 8


class AllLayersEmbedder:
    """
    Embeds texts using the last-token representation from every transformer layer,
    including the input embedding layer (layer 0).

    embed_all_layers() returns a list of np.ndarray, one per layer,
    each of shape (N, H).
    """

    def __init__(self, cfg: EmbedConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(
            cfg.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map=None,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.input_device = next(self.model.parameters()).device

        # Detect number of layers (N transformer blocks → N+1 hidden states with embedding layer)
        cfg_model = self.model.config
        self.num_hidden_layers = getattr(cfg_model, "num_hidden_layers", None)
        if self.num_hidden_layers is not None:
            # +1 for the embedding layer (layer 0)
            self.num_layers = self.num_hidden_layers + 1
        else:
            self.num_layers = None  # determined at runtime from first batch

    @torch.no_grad()
    def embed_all_layers(self, texts: List[str]) -> List[np.ndarray]:
        """
        Returns a list of arrays, one per layer (including embedding layer).
        Each array has shape (N, H).
        """
        bs = self.cfg.batch_size
        # Accumulate per-layer vectors: layer_vecs[layer_idx] = list of batch arrays
        layer_vecs: Optional[List[List[np.ndarray]]] = None

        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )
            input_ids = enc["input_ids"].to(self.input_device)
            attention_mask = enc["attention_mask"].to(self.input_device)

            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            # hidden_states: tuple of (num_layers+1) tensors, each (batch, seq_len, H)
            # index 0 = token embeddings, index k = after transformer block k
            hidden_states = out.hidden_states  # tuple length = num_hidden_layers + 1

            if layer_vecs is None:
                layer_vecs = [[] for _ in range(len(hidden_states))]

            lengths = attention_mask.sum(dim=1)
            last_token_idx = (lengths - 1).clamp(min=0)

            for layer_i, hs in enumerate(hidden_states):
                # hs: (batch, seq_len, H)
                batch_vecs = hs[
                    torch.arange(hs.size(0), device=hs.device),
                    last_token_idx,
                    :
                ]
                layer_vecs[layer_i].append(batch_vecs.detach().float().cpu().numpy())

        if layer_vecs is None:
            raise ValueError("No texts provided to embed.")

        return [np.concatenate(vecs, axis=0) for vecs in layer_vecs]


# -----------------------
# Cache helpers
# -----------------------

def save_all_layers_npz(
    out_dir: str,
    X_layers: List[np.ndarray],
    y_all: np.ndarray,
    meta: Dict[str, Any],
    texts_all: Optional[List[str]] = None,
    all_original_indices: Optional[List[int]] = None,
):
    """
    Save all-layer embeddings.
    X_layers[i] has shape (N, H_i).
    Stored as layer_0, layer_1, ... keys inside the npz.
    """
    os.makedirs(out_dir, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {}
    for i, X in enumerate(X_layers):
        arrays[f"layer_{i}"] = X.astype(np.float32)

    y_all = np.asarray(y_all)
    if np.issubdtype(y_all.dtype, np.floating):
        arrays["y_all"] = y_all.astype(np.float32)
        meta["y_dtype"] = "float32"
    else:
        arrays["y_all"] = y_all.astype(np.int64)
        meta["y_dtype"] = "int64"

    meta["num_layers"] = len(X_layers)
    np.savez_compressed(os.path.join(out_dir, "all_layers_dataset.npz"), **arrays)

    with open(os.path.join(out_dir, "all_layers_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if texts_all is not None or all_original_indices is not None:
        texts_payload = {
            "texts_all": texts_all or [],
            "all_original_indices": all_original_indices or [],
        }
        with open(os.path.join(out_dir, "texts.json"), "w", encoding="utf-8") as f:
            json.dump(texts_payload, f, ensure_ascii=False, indent=2)


def load_all_layers_npz(out_dir: str):
    """
    Load all-layer embeddings saved by save_all_layers_npz.
    Returns (X_layers, y_all, texts_all, all_original_indices) or None.
    """
    path = os.path.join(out_dir, "all_layers_dataset.npz")
    if not os.path.exists(path):
        return None

    data = np.load(path)
    meta_path = os.path.join(out_dir, "all_layers_meta.json")
    y_dtype = "int64"
    num_layers = None
    if os.path.exists(meta_path):
        with open(meta_path) as mf:
            meta = json.load(mf)
        y_dtype = meta.get("y_dtype", "int64")
        num_layers = meta.get("num_layers", None)

    y_all = data["y_all"].astype(y_dtype)

    if num_layers is None:
        layer_keys = sorted([k for k in data.files if k.startswith("layer_")],
                            key=lambda k: int(k.split("_")[1]))
    else:
        layer_keys = [f"layer_{i}" for i in range(num_layers)]

    X_layers = [data[k] for k in layer_keys]

    texts_all = None
    all_original_indices = None
    texts_path = os.path.join(out_dir, "texts.json")
    if os.path.exists(texts_path):
        with open(texts_path, encoding="utf-8") as tf:
            tp = json.load(tf)
        texts_all = tp.get("texts_all") or None
        all_original_indices = tp.get("all_original_indices") or None

    return X_layers, y_all, texts_all, all_original_indices


# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser(
        description="Search all transformer layers for the best embedding layer for classification."
    )
    ap.add_argument("--results_path", type=str, required=True)
    ap.add_argument("--label_path", type=str, required=True)
    ap.add_argument(
        "--data_root",
        type=str,
        default="/NS/chatgpt/work/qwu/hallucinations_detection/data/tool_predictor"
    )
    ap.add_argument(
        "--setting",
        type=str,
        default="auto",
        choices=["auto", "no_search", "with_search", "unknown"]
    )
    ap.add_argument(
        "--model_name_or_path",
        type=str,
        default="auto",
        help="If auto, uses item['model'] from the first row that has it."
    )
    ap.add_argument(
        "--predictor_id",
        type=int,
        default=1,
        help="Single integer predictor id, e.g. 1 or 2"
    )
    ap.add_argument(
        "--classifiers",
        type=str,
        default="LogisticRegression,MLP,XGBoost",
        help="Comma list of classifiers to run. Choices: LogisticRegression, MLP, XGBoost"
    )
    ap.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of folds for cross-validation (default: 5)"
    )
    ap.add_argument(
        "--selection_metric",
        type=str,
        default="oof_roc_auc",
        choices=["oof_roc_auc", "mean_fold_roc_auc", "balanced_accuracy", "accuracy"],
        help="Metric used to select the best layer (default: oof_roc_auc)"
    )
    ap.add_argument(
        "--selection_classifier",
        type=str,
        default="best",
        help=(
            "Which classifier's metric to use for layer selection. "
            "'best' picks the max across all classifiers per layer."
        )
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument(
        "--force_rebuild",
        action="store_true",
        help="Re-embed and rebuild all-layers dataset even if already cached."
    )
    ap.add_argument(
        "--layers",
        type=str,
        default="all",
        help=(
            "Which layers to evaluate. 'all' = every layer (default). "
            "Or a comma-separated list of layer indices, e.g. '0,8,16,32'."
        )
    )
    args = ap.parse_args()

    set_seed(args.seed)

    rows = read_jsonl(args.results_path)
    if not rows:
        raise ValueError(f"No rows found in {args.results_path}")

    setting = (
        detect_setting_from_filename(args.results_path)
        if args.setting == "auto"
        else args.setting
    )

    model_name_or_path = args.model_name_or_path
    if model_name_or_path == "auto":
        model_name_or_path = None
        for r in rows:
            if isinstance(r.get("model", None), str) and r["model"]:
                model_name_or_path = r["model"]
                break
        if model_name_or_path is None:
            raise ValueError(
                "Could not infer model path from item['model']. "
                "Provide --model_name_or_path explicitly."
            )
    print(f"Parsed model as: {model_name_or_path}")
    model_dirname = sanitize_model_name(model_name_or_path)

    predictor_id = args.predictor_id
    clf_names = [c.strip() for c in args.classifiers.split(",") if c.strip()]

    labels, valid_indices, label_scores = read_labels_from_path(
        predictor_id,
        args.label_path,
        score_threshold_1=0.9,
        score_threshold_2=0.1,
    )

    if not valid_indices:
        raise ValueError("No valid samples with labels found.")

    if len(rows) <= max(valid_indices):
        raise ValueError(
            f"Label file has index {max(valid_indices)} but results file only has {len(rows)} rows."
        )

    out_dir = os.path.join(args.data_root, model_dirname, f"predictor_{predictor_id}_layer_search")
    os.makedirs(out_dir, exist_ok=True)

    all_layers_path = os.path.join(out_dir, "all_layers_dataset.npz")
    need_embedder = args.force_rebuild or not os.path.exists(all_layers_path)

    if need_embedder:
        embedder = AllLayersEmbedder(
            EmbedConfig(
                model_name_or_path=model_name_or_path,
                device=args.device,
                batch_size=args.batch_size,
            )
        )

        texts_by_idx, labels_by_idx = build_dataset_by_original_index(
            rows=rows,
            valid_indices=valid_indices,
            labels=labels,
            setting=setting,
            predictor_id=predictor_id,
        )

        all_original_indices = sorted(texts_by_idx.keys())

        if len(all_original_indices) < args.n_splits:
            raise ValueError(
                f"Too few samples ({len(all_original_indices)}) for {args.n_splits}-fold CV "
                f"in predictor_{predictor_id} after filtering."
            )

        texts_all = [texts_by_idx[i] for i in all_original_indices]
        y_all = np.array([labels_by_idx[i] for i in all_original_indices], dtype=np.int64)

        print(f"\nEmbedding {len(texts_all)} samples across all layers ...")
        X_layers = embedder.embed_all_layers(texts_all)
        print(f"Extracted {len(X_layers)} layers, each shape {X_layers[0].shape}")

        meta = {
            "results_path": args.results_path,
            "label_path": args.label_path,
            "setting": setting,
            "model_name_or_path": model_name_or_path,
            "predictor_id": predictor_id,
            "num_samples": int(len(y_all)),
            "label_counts": {
                str(k): int(v)
                for k, v in zip(*np.unique(y_all, return_counts=True))
            },
            "score_threshold": 0.9,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "all_original_indices": list(map(int, all_original_indices)),
        }
        save_all_layers_npz(
            out_dir,
            X_layers,
            y_all,
            meta,
            texts_all=texts_all,
            all_original_indices=all_original_indices,
        )
        print(f"Saved all-layer dataset to: {out_dir}")

    else:
        loaded = load_all_layers_npz(out_dir)
        if loaded is None:
            raise ValueError(f"Found {all_layers_path} but failed to load it.")
        X_layers, y_all, texts_all, all_original_indices = loaded
        print(f"Loaded {len(X_layers)} cached layers, each shape {X_layers[0].shape}")

    # Determine which layers to evaluate
    if args.layers == "all":
        layer_indices = list(range(len(X_layers)))
    else:
        layer_indices = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
        invalid = [i for i in layer_indices if i >= len(X_layers)]
        if invalid:
            raise ValueError(f"Requested layer indices {invalid} out of range (0–{len(X_layers)-1})")

    print(f"\n{'=' * 60}")
    print(f"  Layer search | model={model_dirname} | predictor={predictor_id}")
    print(f"  Layers to evaluate: {layer_indices}")
    print(f"  Selection metric: {args.selection_metric} | classifier: {args.selection_classifier}")
    print(f"{'=' * 60}")

    all_layer_scores: Dict[int, Dict[str, Any]] = {}

    for layer_i in layer_indices:
        X = X_layers[layer_i]
        print(f"\n{'─' * 60}")
        print(f"  Layer {layer_i}/{len(X_layers)-1}  (shape={X.shape})")
        print(f"{'─' * 60}")

        layer_out_dir = os.path.join(out_dir, f"layer_{layer_i}")
        scores_path = os.path.join(layer_out_dir, "scores.json")

        # Re-use cached layer scores if available and not force_rebuild
        if not args.force_rebuild and os.path.exists(scores_path):
            print(f"  [cache] Loading existing scores from {scores_path}")
            with open(scores_path) as f:
                layer_scores = json.load(f)
        else:
            layer_scores = train_and_report(
                X,
                y_all,
                n_splits=args.n_splits,
                clf_names=clf_names,
                out_dir=layer_out_dir,
                texts_all=texts_all,
                all_original_indices=all_original_indices,
                seed=args.seed,
                layer_idx=layer_i,
            )
            os.makedirs(layer_out_dir, exist_ok=True)
            with open(scores_path, "w") as f:
                json.dump(layer_scores, f, indent=2)
            print(f"  Layer {layer_i} scores saved to: {scores_path}")

        all_layer_scores[layer_i] = layer_scores

    # -----------------------
    # Find best layer
    # -----------------------
    metric = args.selection_metric
    sel_clf = args.selection_classifier

    def _layer_score(layer_scores_dict: Dict[str, Any]) -> float:
        """Return scalar score for a layer given classifier results."""
        values = []
        for clf_name, clf_data in layer_scores_dict.items():
            if sel_clf != "best" and clf_name != sel_clf:
                continue
            v = clf_data["result"].get(metric, None)
            if v is not None:
                values.append(float(v))
        return max(values) if values else -1.0

    scored_layers = [(li, _layer_score(all_layer_scores[li])) for li in layer_indices]
    scored_layers.sort(key=lambda x: x[1], reverse=True)

    best_layer_idx, best_score = scored_layers[0]
    best_layer_clf_scores = {
        clf: all_layer_scores[best_layer_idx][clf]["result"].get(metric)
        for clf in clf_names
        if clf in all_layer_scores[best_layer_idx]
    }

    print(f"\n{'=' * 60}")
    print(f"  LAYER SEARCH COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Best layer : {best_layer_idx}  ({metric}={best_score:.4f})")
    print(f"  Classifier scores at best layer:")
    for clf, s in best_layer_clf_scores.items():
        print(f"    {clf}: {metric}={s}")
    print(f"\n  Top 5 layers by {metric}:")
    for rank, (li, sc) in enumerate(scored_layers[:5], 1):
        print(f"    #{rank}  layer {li:3d}  {metric}={sc:.4f}")

    # -----------------------
    # Save summary
    # -----------------------
    summary = {
        "model_name_or_path": model_name_or_path,
        "predictor_id": predictor_id,
        "selection_metric": metric,
        "selection_classifier": sel_clf,
        "best_layer": best_layer_idx,
        "best_score": round(best_score, 4),
        "best_layer_clf_scores": best_layer_clf_scores,
        "ranked_layers": [
            {"layer": li, metric: round(sc, 4)} for li, sc in scored_layers
        ],
        "all_layer_scores": {
            str(li): {
                clf: {
                    "oof_roc_auc": s["result"].get("oof_roc_auc"),
                    "mean_fold_roc_auc": s["result"].get("mean_fold_roc_auc"),
                    "accuracy": s["result"].get("accuracy"),
                    "balanced_accuracy": s["result"].get("balanced_accuracy"),
                }
                for clf, s in layer_scores.items()
            }
            for li, layer_scores in all_layer_scores.items()
        },
    }

    summary_path = os.path.join(out_dir, "layer_search_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
