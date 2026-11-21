"""Train the Dual ML Pipeline (Algorithm Matcher + Bug Root-Cause Classifier) using XGBoost & PyTorch."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.dataset_builder import DEFAULT_DATASET_PATH, build_training_dataset
from src.feature_extractor import (
    ALGO_FAMILIES,
    BUG_CLASSES,
    FEATURE_NAMES,
    extract_all_features,
)
from src.ml_predictor import DEFAULT_WEIGHTS_PATH
from src.rag_retriever import tokenize
from src.real_problems_catalog import ensure_real_catalog_built

ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / "assets"
ARTIFACTS_DIR = ROOT_DIR / "data" / "models"


def _train_naive_bayes_algo_model(records: list[dict[str, Any]], max_vocab: int = 450) -> dict[str, Any]:
    """Train a fast Multinomial Naive Bayes classifier on problem statements + code tokens."""
    classes = list(ALGO_FAMILIES)
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    token_counts: Counter[str] = Counter()
    docs_tokens: list[tuple[int, list[str]]] = []

    for r in records:
        c_idx = cls_to_idx[r["algo_family"]]
        toks = tokenize(f"{r['problem_statement']} {r['code']}")
        token_counts.update(toks)
        docs_tokens.append((c_idx, toks))

    vocab = {tok: idx for idx, (tok, _) in enumerate(token_counts.most_common(max_vocab))}
    n_classes = len(classes)
    n_vocab = len(vocab)

    class_doc_counts = np.ones(n_classes, dtype=np.float64)
    word_counts = np.ones((n_classes, n_vocab), dtype=np.float64)  # Laplace smoothing alpha=1.0

    for c_idx, toks in docs_tokens:
        class_doc_counts[c_idx] += 1.0
        for t in toks:
            v_idx = vocab.get(t)
            if v_idx is not None:
                word_counts[c_idx, v_idx] += 1.0

    class_log_priors = np.log(class_doc_counts / np.sum(class_doc_counts))
    feature_log_prob = np.log(word_counts / np.sum(word_counts, axis=1, keepdims=True))

    # Evaluate accuracy on training set
    correct = 0
    for c_idx, toks in docs_tokens:
        x = np.zeros(n_vocab, dtype=np.float64)
        for t in toks:
            v_idx = vocab.get(t)
            if v_idx is not None:
                x[v_idx] += 1.0
        pred = int(np.argmax(class_log_priors + (feature_log_prob @ x)))
        if pred == c_idx:
            correct += 1

    acc = round(correct / max(len(docs_tokens), 1), 4)
    return {
        "classes": classes,
        "vocab": vocab,
        "class_log_priors": np.round(class_log_priors, 5).tolist(),
        "feature_log_prob": np.round(feature_log_prob, 5).tolist(),
        "train_accuracy": acc,
    }


def run_training_pipeline(samples_per_class: int = 400, seed: int = 42) -> dict[str, Any]:
    """Build dataset, extract 33 AST+Constraint features, run 5-Fold CV with XGBoost & PyTorch, and export JSON weights."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    t0 = time.perf_counter()
    print("[0/5] Construindo catalogo real de 3.500+ questoes do Codeforces com casos de teste oficiais...")
    catalog = ensure_real_catalog_built()
    print(f"    -> Catalogo real pronto: {len(catalog):,} questoes indexadas em data/real_problems_catalog.json.gz")
    print("[1/5] Extraindo e consolidando submissoes reais (DeepMind CodeContests + IBM CodeNet4Repair + Codeforces)...")
    records = build_training_dataset(DEFAULT_DATASET_PATH, samples_per_class=samples_per_class, seed=seed)

    print("[2/5] Extraindo 33 features AST + Restricoes Matematicas dos enunciados...")
    cls_to_idx = {c: i for i, c in enumerate(BUG_CLASSES)}
    X_list: list[list[float]] = []
    y_list: list[int] = []

    for r in records:
        analysis = extract_all_features(
            code=r["code"],
            language=r["language"],
            problem_statement=r["problem_statement"],
        )
        X_list.append(analysis.features)
        y_list.append(cls_to_idx[r["bug_class"]])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[3/5] Treinando Modelo 1 (Algorithm & Problem Matcher) e Modelo 2 (XGBoost vs PyTorch MLP em device={device})...")
    algo_model = _train_naive_bayes_algo_model(records)

    class BugMLP(nn.Module):
        def __init__(self, in_dim: int, num_classes: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(in_dim, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, num_classes)
            self.relu = nn.ReLU()

        def forward(self, x_in: torch.Tensor) -> torch.Tensor:
            h1 = self.relu(self.fc1(x_in))
            h2 = self.relu(self.fc2(h1))
            return self.fc3(h2)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    xgb_oof = np.zeros_like(y)
    mlp_oof = np.zeros_like(y)
    epoch_loss_history: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va = X[val_idx]

        # 1. XGBoost Multi-class
        xgb = XGBClassifier(
            n_estimators=140,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed + fold,
            n_jobs=-1,
            eval_metric="mlogloss",
        )
        xgb.fit(X_tr, y_tr)
        xgb_oof[val_idx] = xgb.predict(X_va)

        # 2. PyTorch BugMLP
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
        X_va_s = scaler.transform(X_va).astype(np.float32)

        torch.manual_seed(seed + fold)
        model = BugMLP(X.shape[1], len(BUG_CLASSES)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        t_x = torch.from_numpy(X_tr_s).to(device)
        t_y = torch.from_numpy(y_tr).to(device)

        model.train()
        for _ in range(70):
            optimizer.zero_grad()
            out = model(t_x)
            loss = criterion(out, t_y)
            loss.backward()
            optimizer.step()
            if fold == 1:
                epoch_loss_history.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            v_x = torch.from_numpy(X_va_s).to(device)
            preds = torch.argmax(model(v_x), dim=1).cpu().numpy()
            mlp_oof[val_idx] = preds

    xgb_f1 = round(float(f1_score(y, xgb_oof, average="macro")), 4)
    xgb_acc = round(float(accuracy_score(y, xgb_oof)), 4)
    mlp_f1 = round(float(f1_score(y, mlp_oof, average="macro")), 4)
    mlp_acc = round(float(accuracy_score(y, mlp_oof)), 4)
    mlp_prec = round(float(precision_score(y, mlp_oof, average="macro")), 4)
    mlp_rec = round(float(recall_score(y, mlp_oof, average="macro")), 4)

    print(f"    -> 5-Fold CV XGBoost: Accuracy={xgb_acc:.4f} | Macro-F1={xgb_f1:.4f}")
    print(f"    -> 5-Fold CV PyTorch BugMLP: Accuracy={mlp_acc:.4f} | Macro-F1={mlp_f1:.4f}")
    print(f"    -> Algorithm Matcher Accuracy: {algo_model['train_accuracy']:.4f}")

    # Fit final full-dataset PyTorch BugMLP for ultra-lightweight JSON weight export (< 500 KB)
    print("[4/5] Exportando pesos finais compactos para a Vercel (Zero-API NumPy Runtime)...")
    final_scaler = StandardScaler()
    X_full_s = final_scaler.fit_transform(X).astype(np.float32)
    torch.manual_seed(seed)
    final_mlp = BugMLP(X.shape[1], len(BUG_CLASSES)).to(device)
    opt = torch.optim.AdamW(final_mlp.parameters(), lr=0.01, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    tx_full = torch.from_numpy(X_full_s).to(device)
    ty_full = torch.from_numpy(y).to(device)

    final_mlp.train()
    for _ in range(85):
        opt.zero_grad()
        loss = crit(final_mlp(tx_full), ty_full)
        loss.backward()
        opt.step()

    final_mlp.eval()
    sd = final_mlp.cpu().state_dict()

    elapsed_sec = round(time.perf_counter() - t0, 2)
    metrics_summary = {
        "dataset_size": len(records),
        "num_features": len(FEATURE_NAMES),
        "num_bug_classes": len(BUG_CLASSES),
        "num_algo_families": len(ALGO_FAMILIES),
        "device_used": str(device),
        "training_time_sec": elapsed_sec,
        "xgboost_cv_accuracy": xgb_acc,
        "xgboost_cv_macro_f1": xgb_f1,
        "pytorch_mlp_cv_accuracy": mlp_acc,
        "pytorch_mlp_cv_macro_f1": mlp_f1,
        "pytorch_mlp_cv_precision": mlp_prec,
        "pytorch_mlp_cv_recall": mlp_rec,
        "algo_classifier_accuracy": algo_model["train_accuracy"],
    }

    export_payload = {
        "metadata": metrics_summary,
        "feature_names": FEATURE_NAMES,
        "bug_classes": list(BUG_CLASSES),
        "bug_mlp": {
            "scaler_mean": np.round(final_scaler.mean_, 6).tolist(),
            "scaler_scale": np.round(final_scaler.scale_, 6).tolist(),
            "w1": np.round(sd["fc1.weight"].numpy(), 5).tolist(),
            "b1": np.round(sd["fc1.bias"].numpy(), 5).tolist(),
            "w2": np.round(sd["fc2.weight"].numpy(), 5).tolist(),
            "b2": np.round(sd["fc2.bias"].numpy(), 5).tolist(),
            "w3": np.round(sd["fc3.weight"].numpy(), 5).tolist(),
            "b3": np.round(sd["fc3.bias"].numpy(), 5).tolist(),
        },
        "algo_model": algo_model,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_WEIGHTS_PATH.write_text(
        json.dumps(export_payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    # Save visual charts in assets/
    print("[5/5] Gerando graficos de convergencia e matriz de confusao em assets/...")
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(range(1, len(epoch_loss_history) + 1), epoch_loss_history, color="#38bdf8", linewidth=2.2)
    axes[0].set_title("PyTorch BugMLP - Cross-Entropy Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    cm = confusion_matrix(y, mlp_oof)
    im = axes[1].imshow(cm, cmap="Blues")
    axes[1].set_title(f"5-Fold CV Confusion Matrix (Macro-F1: {mlp_f1:.3f})")
    short_labels = ["Overflow", "TLE O(N²)", "OffByOne", "GlobalDirty", "SlowIO", "InfLoop/Rec", "Edge/Uninit", "Clean"]
    axes[1].set_xticks(range(len(short_labels)))
    axes[1].set_yticks(range(len(short_labels)))
    axes[1].set_xticklabels(short_labels, rotation=35, ha="right", fontsize=8)
    axes[1].set_yticklabels(short_labels, fontsize=8)
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(ASSETS_DIR / "training_metrics.png", dpi=140)
    plt.close(fig)

    print(f"Concluido em {elapsed_sec}s! Pesos salvos em: {DEFAULT_WEIGHTS_PATH}")
    return metrics_summary


if __name__ == "__main__":
    run_training_pipeline()
