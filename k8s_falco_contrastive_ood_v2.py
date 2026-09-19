#!/usr/bin/env python3
"""
Kubernetes/Falco Leakage-Resistant Unseen-Attack Detection
========================================
Self-Supervised Contrastive Transformer + Open-Set/OOD Detection
with Leave-One-Rule-Out evaluation.

Research purpose:
- Train WITHOUT one attack rule/family.
- Learn Falco-event representations using contrastive self-supervision.
- Fine-tune a known benign/attack classifier.
- Fit class-conditional Mahalanobis OOD statistics on learned embeddings.
- Detect the held-out rule as UNKNOWN / potential unseen attack.

Expected CSV:
- one text-like event column (e.g. raw_line, output, message, alert, event)
- one binary label column (e.g. label, class, target)
- one attack/rule column (e.g. rule, rule_name, attack_type)
- timestamp optional.

The script can auto-detect common column names, or you can pass them explicitly.

Example:
python k8s_falco_contrastive_ood.py \
    --csv /data/PyCharmMiscProject/WARP_Falco_Alerts_Labeled_Dataset.csv \
    --label-col label \
    --rule-col rule \
    --text-col raw_line \
    --heldout-rule "Terminal shell in container" \
    --epochs-ssl 5 \
    --epochs-cls 5 \
    --output-dir /data/PyCharmMiscProject/k8s_ood_results

To discover rules only:
python k8s_falco_contrastive_ood.py --csv FILE.csv --list-rules

To automatically test the top N attack rules:
python k8s_falco_contrastive_ood.py --csv FILE.csv --auto-holdouts 5
"""

import os
import re
import json
import math
import random
import argparse
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    average_precision_score, confusion_matrix
)
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------------------------
# Column inference
# ---------------------------------------------------------------------

COLUMN_CANDIDATES = {
    "label": ["label", "class", "target", "is_attack", "attack_label", "binary_label"],
    "rule": [
        "rule", "rule_name", "falco_rule", "attack_type",
        "attack", "event_type", "signature"
    ],
    "text": [
        "raw_line", "raw", "output", "message", "alert", "alert_text",
        "event", "log", "description", "falco_output"
    ],
    "timestamp": [
        "timestamp", "time", "datetime", "event_time", "ts"
    ]
}

def infer_col(df, kind, explicit=None, required=True):
    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(
                f"--{kind}-col='{explicit}' not found. Available columns:\n"
                + ", ".join(map(str, df.columns))
            )
        return explicit

    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in COLUMN_CANDIDATES[kind]:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    if required:
        raise ValueError(
            f"Could not auto-detect {kind} column. "
            f"Pass --{kind}-col explicitly.\nAvailable columns:\n"
            + ", ".join(map(str, df.columns))
        )
    return None

# ---------------------------------------------------------------------
# Binary label normalization
# ---------------------------------------------------------------------

NORMAL_WORDS = {
    "normal", "benign", "0", "false", "clean", "legitimate",
    "non-attack", "non_attack", "safe"
}
ATTACK_WORDS = {
    "attack", "malicious", "1", "true", "anomaly", "intrusion",
    "abnormal", "malware"
}

def normalize_binary_label(series):
    # Numeric 0/1
    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce")
        uniq = set(vals.dropna().unique().tolist())
        if uniq.issubset({0, 1, 0.0, 1.0}):
            return vals.fillna(0).astype(int)

    s = series.astype(str).str.strip().str.lower()

    out = []
    unknown_examples = []
    for x in s:
        if x in NORMAL_WORDS:
            out.append(0)
        elif x in ATTACK_WORDS:
            out.append(1)
        else:
            # heuristic
            if any(k in x for k in ["attack", "malic", "intrus", "anomal"]):
                out.append(1)
            elif any(k in x for k in ["normal", "benign", "legit", "clean"]):
                out.append(0)
            else:
                unknown_examples.append(x)
                out.append(None)

    if unknown_examples:
        examples = sorted(set(unknown_examples))[:20]
        raise ValueError(
            "Could not map some label values to binary benign/attack. "
            f"Examples: {examples}\n"
            "Please convert the label column to 0=benign, 1=attack first."
        )
    return pd.Series(out, index=series.index).astype(int)

# ---------------------------------------------------------------------
# Leakage-resistant Falco/Kubernetes sanitization
# ---------------------------------------------------------------------

UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
LONG_HEX_RE = re.compile(r"\b[0-9a-fA-F]{10,64}\b")
FALCO_TIME_PREFIX_RE = re.compile(
    r"^\s*(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?:\s*"
)
K8S_POD_RE = re.compile(r"\bk8s\.pod=([^\s\)]+)")
CONTAINER_FIELD_RE = re.compile(r"\bcontainer=([0-9a-fA-F]{8,64})\b")
COMMAND_CONTAINER_RE = re.compile(r"\bcommand=container:([0-9a-fA-F]{8,64})\b")
# Common standalone Kubernetes pod/deployment names:
# examples:
# falco-event-generator-f7rlt
# privileged-deployment-574878fc9d-cmdkr
# sensitive-mount-deployment-7fd6f8467-g25nz
STANDALONE_POD_RE = re.compile(
    r"\b(?:"
    r"[a-z0-9][a-z0-9-]*-[0-9a-f]{8,10}-[a-z0-9]{5}"
    r"|"
    r"[a-z0-9][a-z0-9-]*-[a-z0-9]{5}"
    r")\b"
)
POD_DEPLOYMENT_SUFFIX_RE = re.compile(
    r"\b([a-zA-Z0-9][a-zA-Z0-9_.-]*?)-[0-9a-f]{8,10}-[a-z0-9]{5}\b"
)
GENERIC_HASH_SUFFIX_RE = re.compile(r"(?<=-)[0-9a-f]{8,10}(?=-|\b)")


def extract_raw_pod_name(text):
    """Extract raw pod identity for splitting only; never feed it to the model."""
    m = K8S_POD_RE.search(str(text))
    return m.group(1) if m else None




def build_group_ids(df, text_col):
    """
    Group by raw Kubernetes pod identity when available.
    Rows with no pod get their own row-level group.
    """
    groups = []
    for idx, text in zip(df.index, df[text_col].astype(str)):
        pod = extract_raw_pod_name(text)
        groups.append(f"POD::{pod}" if pod else f"ROW::{idx}")
    return pd.Series(groups, index=df.index)

# ---------------------------------------------------------------------
# Simple tokenizer
# ---------------------------------------------------------------------
def sanitize_falco_text(text):
    """
    Remove environment-specific identifiers while preserving
    security-relevant behavioral information.
    """

    x = str(text)

    # Normalize escaped/newline characters
    x = x.replace("\\n", " ")
    x = x.replace("\n", " ")

    # Remove full Falco timestamp
    x = FALCO_TIME_PREFIX_RE.sub("", x)

    # Normalize container-related fields
    x = COMMAND_CONTAINER_RE.sub(
        "command=CONTAINER",
        x
    )

    x = CONTAINER_FIELD_RE.sub(
        "container=CONTAINER",
        x
    )

    # Extract exact Kubernetes pod names before removing them.
    # Then replace those exact pod identifiers everywhere in the alert.
    pod_names = K8S_POD_RE.findall(x)

    for pod_name in set(pod_names):
        x = re.sub(
            rf"\b{re.escape(pod_name)}\b",
            "POD",
            x
        )

    # Normalize any remaining k8s.pod field
    x = K8S_POD_RE.sub(
        "k8s.pod=POD",
        x
    )

    # Replace UUIDs
    x = UUID_RE.sub(
        "UUID",
        x
    )

    # Replace IP addresses
    x = IPV4_RE.sub(
        "IPADDR",
        x
    )

    # Normalize deployment/pod hashes
    x = POD_DEPLOYMENT_SUFFIX_RE.sub(
        r"\1-POD",
        x
    )

    x = GENERIC_HASH_SUFFIX_RE.sub(
        "HASH",
        x
    )

    # Replace remaining long hexadecimal identifiers
    x = LONG_HEX_RE.sub(
        "HEXID",
        x
    )

    # Normalize container_id field if still present
    x = re.sub(
        r"\bcontainer_id=[^\s\)]+",
        "container_id=CONTAINER",
        x
    )

    # Normalize whitespace
    x = re.sub(
        r"\s+",
        " ",
        x
    ).strip()

    return x
TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_./:-]*|\d+\.\d+\.\d+\.\d+|\d+|[^\s]"
)

def tokenize(text):
    return TOKEN_RE.findall(str(text).lower())

class SimpleVocab:
    PAD = "<PAD>"
    CLS = "<CLS>"
    UNK = "<UNK>"

    def __init__(self, max_size=30000, min_freq=2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.itos = [self.PAD, self.CLS, self.UNK]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def fit(self, texts):
        cnt = Counter()
        for t in texts:
            cnt.update(tokenize(t))
        for tok, freq in cnt.most_common():
            if freq < self.min_freq:
                break
            if len(self.itos) >= self.max_size:
                break
            if tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

    def encode(self, text, max_len):
        toks = tokenize(text)
        ids = [self.stoi[self.CLS]]
        ids += [self.stoi.get(t, self.stoi[self.UNK]) for t in toks[:max_len-1]]
        if len(ids) < max_len:
            ids += [self.stoi[self.PAD]] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.itos)

# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class FalcoDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=96, augment=False, token_drop=0.12):
        self.texts = list(map(str, texts))
        self.labels = np.asarray(labels, dtype=np.int64)
        self.vocab = vocab
        self.max_len = max_len
        self.augment = augment
        self.token_drop = token_drop
        self.pad_id = vocab.stoi[vocab.PAD]
        self.cls_id = vocab.stoi[vocab.CLS]
        self.unk_id = vocab.stoi[vocab.UNK]

    def __len__(self):
        return len(self.texts)

    def _aug(self, ids):
        ids = list(ids)
        # preserve CLS; randomly mask/drop content tokens to UNK.
        for i in range(1, len(ids)):
            if ids[i] == self.pad_id:
                break
            if random.random() < self.token_drop:
                ids[i] = self.unk_id
        return ids

    def __getitem__(self, idx):
        ids = self.vocab.encode(self.texts[idx], self.max_len)
        y = int(self.labels[idx])
        if self.augment:
            v1 = torch.tensor(self._aug(ids), dtype=torch.long)
            v2 = torch.tensor(self._aug(ids), dtype=torch.long)
            return v1, v2, torch.tensor(y, dtype=torch.long)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# ---------------------------------------------------------------------
# Transformer model
# ---------------------------------------------------------------------

class FalcoTransformer(nn.Module):
    def __init__(
        self, vocab_size, max_len=96, d_model=128, nhead=4,
        num_layers=2, dim_ff=256, dropout=0.1, proj_dim=128
    ):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim)
        )
        self.classifier = nn.Linear(d_model, 2)

    def encode(self, x):
        b, l = x.shape
        pos = torch.arange(l, device=x.device).unsqueeze(0).expand(b, l)
        h = self.tok_emb(x) * math.sqrt(self.d_model) + self.pos_emb(pos)
        pad_mask = (x == 0)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        cls = self.norm(h[:, 0, :])
        return cls

    def forward(self, x):
        z = self.encode(x)
        logits = self.classifier(z)
        proj = F.normalize(self.projector(z), dim=-1)
        return logits, proj, z

# ---------------------------------------------------------------------
# Contrastive learning
# ---------------------------------------------------------------------

def nt_xent_loss(z1, z2, temperature=0.15):
    """
    SimCLR-style NT-Xent loss.
    Positive pair: two augmented views of the same Falco event.
    FP32 is enforced here for numerical stability under CUDA AMP.
    """
    b = z1.size(0)

    z1 = z1.float()
    z2 = z2.float()

    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature

    mask = torch.eye(2 * b, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, torch.finfo(sim.dtype).min)

    pos_idx = torch.arange(2 * b, device=z.device)
    pos_idx = (pos_idx + b) % (2 * b)

    return F.cross_entropy(sim, pos_idx)

def train_ssl(model, loader, optimizer, device, epochs=5, temperature=0.15):
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    for ep in range(1, epochs + 1):
        losses = []
        for v1, v2, _ in loader:
            v1, v2 = v1.to(device), v2.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                _, p1, _ = model(v1)
                _, p2, _ = model(v2)
                loss = nt_xent_loss(p1, p2, temperature)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        print(f"[SSL] epoch {ep:02d}/{epochs} loss={np.mean(losses):.5f}")

def train_classifier(model, loader, optimizer, device, epochs=5):
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(1, epochs + 1):
        losses, preds, ys = [], [], []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits, _, _ = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.append(float(loss.detach().cpu()))
            preds.extend(logits.argmax(1).detach().cpu().numpy().tolist())
            ys.extend(y.detach().cpu().numpy().tolist())

        acc = accuracy_score(ys, preds)
        print(f"[CLS] epoch {ep:02d}/{epochs} loss={np.mean(losses):.5f} acc={acc:.4f}")

# ---------------------------------------------------------------------
# Embeddings / predictions
# ---------------------------------------------------------------------

@torch.no_grad()
def infer_embeddings(model, loader, device):
    model.eval()
    emb, logits_all, ys = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits, _, z = model(x)
        emb.append(z.cpu().numpy())
        logits_all.append(logits.cpu().numpy())
        ys.append(y.numpy())
    return (
        np.concatenate(emb, axis=0),
        np.concatenate(logits_all, axis=0),
        np.concatenate(ys, axis=0)
    )

# ---------------------------------------------------------------------
# Mahalanobis OOD
# ---------------------------------------------------------------------

class MahalanobisOOD:
    """
    Class-conditional Mahalanobis distance.
    Score = minimum distance to any known class centroid.
    Higher = more OOD.
    """
    def __init__(self, shrinkage=1e-3):
        self.means = {}
        self.precision = None
        self.shrinkage = shrinkage

    def fit(self, X, y):
        classes = np.unique(y)
        for c in classes:
            self.means[int(c)] = X[y == c].mean(axis=0)

        cov = np.cov(X, rowvar=False)
        cov += np.eye(cov.shape[0]) * self.shrinkage
        self.precision = np.linalg.pinv(cov)

    def score(self, X):
        all_d = []
        for c, mu in self.means.items():
            d = X - mu
            d2 = np.einsum("bi,ij,bj->b", d, self.precision, d)
            all_d.append(d2)
        return np.min(np.stack(all_d, axis=1), axis=1)

# ---------------------------------------------------------------------
# OOD metrics
# ---------------------------------------------------------------------

def fpr_at_95_tpr(y_ood, scores):
    # y_ood: 1=OOD, 0=ID. Higher score=more OOD.
    y_ood = np.asarray(y_ood)
    scores = np.asarray(scores)
    ood_scores = scores[y_ood == 1]
    id_scores = scores[y_ood == 0]
    if len(ood_scores) == 0 or len(id_scores) == 0:
        return float("nan")
    threshold = np.percentile(ood_scores, 5)  # 95% TPR on OOD
    return float(np.mean(id_scores >= threshold))

def safe_auc(y, scores):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))

def safe_aupr(y, scores):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, scores))

# ---------------------------------------------------------------------
# Split + experiment
# ---------------------------------------------------------------------

def _labels_present(frame):
    return frame["_y"].nunique() >= 2


def _group_split_known(known, group_col="_group_id", seed=42, max_attempts=30):
    """Try leakage-resistant pod/group split; fall back only if necessary."""
    from sklearn.model_selection import GroupShuffleSplit

    for attempt in range(max_attempts):
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed + attempt)
        trainval_idx, test_idx = next(gss1.split(known, groups=known[group_col]))
        trainval = known.iloc[trainval_idx].copy()
        test_known = known.iloc[test_idx].copy()

        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed + 1000 + attempt)
        tr_idx, val_idx = next(gss2.split(trainval, groups=trainval[group_col]))
        train_df = trainval.iloc[tr_idx].copy()
        val_df = trainval.iloc[val_idx].copy()

        if _labels_present(train_df) and _labels_present(val_df) and _labels_present(test_known):
            tr = set(train_df[group_col])
            va = set(val_df[group_col])
            te = set(test_known[group_col])
            if not (tr & va or tr & te or va & te):
                return train_df, val_df, test_known, "group"

    print("[WARNING] Group split could not preserve both classes in every partition; using stratified fallback.")
    trainval, test_known = train_test_split(
        known, test_size=0.20, random_state=seed, stratify=known["_y"]
    )
    train_df, val_df = train_test_split(
        trainval, test_size=0.15, random_state=seed, stratify=trainval["_y"]
    )
    return train_df, val_df, test_known, "stratified_fallback"


def make_splits(df, label_col, rule_col, heldout_rule, seed=42):
    """
    Held-out attack rule is completely absent from train/validation.
    Known data is split by raw pod/group identity whenever feasible.
    """
    is_holdout = (
        (df[rule_col].astype(str) == str(heldout_rule)) &
        (df["_y"] == 1)
    )

    heldout = df[is_holdout].copy()
    known = df[~is_holdout].copy()

    if len(heldout) < 10:
        raise ValueError(
            f"Held-out rule '{heldout_rule}' has only {len(heldout)} attack rows. "
            "Choose a rule with at least 10 attack events."
        )

    train_df, val_df, test_known, split_mode = _group_split_known(
        known, group_col="_group_id", seed=seed
    )

    test_df = pd.concat([test_known, heldout], ignore_index=True)
    test_df["_is_ood"] = 0
    test_df.loc[
        (test_df[rule_col].astype(str) == str(heldout_rule)) &
        (test_df["_y"] == 1),
        "_is_ood"
    ] = 1

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        split_mode,
    )


def threshold_sweep(val_scores, test_scores, y_ood,
                    quantiles=(0.95, 0.975, 0.99, 0.995, 0.999)):
    rows = []
    for q in quantiles:
        thr = float(np.quantile(val_scores, q))
        pred = (test_scores > thr).astype(int)
        tp = int(((pred == 1) & (y_ood == 1)).sum())
        fn = int(((pred == 0) & (y_ood == 1)).sum())
        fp = int(((pred == 1) & (y_ood == 0)).sum())
        tn = int(((pred == 0) & (y_ood == 0)).sum())
        rows.append({
            "quantile": q,
            "threshold": thr,
            "tp_unknown": tp,
            "fp_unknown": fp,
            "tn_id": tn,
            "fn_unknown": fn,
            "unknown_precision": tp / max(tp + fp, 1),
            "unknown_recall": tp / max(tp + fn, 1),
            "id_false_unknown_rate": fp / max(fp + tn, 1),
        })
    return pd.DataFrame(rows)

def top_attack_rules(df, rule_col, n=5, min_rows=50):
    g = (
        df[df["_y"] == 1]
        .groupby(rule_col)
        .size()
        .sort_values(ascending=False)
    )
    g = g[g >= min_rows]
    return [str(x) for x in g.head(n).index.tolist()]

def run_one(args, df, label_col, rule_col, text_col, heldout_rule):
    print("\n" + "="*90)
    print(f"HELD-OUT UNSEEN RULE: {heldout_rule}")
    print("="*90)

    train_df, val_df, test_df, split_mode = make_splits(
        df, label_col, rule_col, heldout_rule, args.seed
    )

    print(f"[SPLIT] mode={split_mode}")
    print(f"[SPLIT] train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")
    print(f"[GROUPS] train={train_df['_group_id'].nunique():,} val={val_df['_group_id'].nunique():,} test={test_df['_group_id'].nunique():,}")
    print(f"[SPLIT] unseen heldout test events={int(test_df['_is_ood'].sum()):,}")

    # Optional training cap for fast experiments.
    if args.max_train_rows and len(train_df) > args.max_train_rows:
        train_df = train_df.sample(args.max_train_rows, random_state=args.seed)
        print(f"[INFO] capped training rows to {len(train_df):,}")

    vocab = SimpleVocab(max_size=args.vocab_size, min_freq=args.min_freq)
    print("[INFO] building vocabulary...")
    vocab.fit(train_df["_sanitized_text"].astype(str).tolist())
    print(f"[INFO] vocabulary size={len(vocab):,}")

    ssl_ds = FalcoDataset(
        train_df["_sanitized_text"], train_df["_y"], vocab,
        max_len=args.max_len, augment=True, token_drop=args.token_drop
    )
    cls_ds = FalcoDataset(
        train_df["_sanitized_text"], train_df["_y"], vocab,
        max_len=args.max_len, augment=False
    )
    val_ds = FalcoDataset(
        val_df["_sanitized_text"], val_df["_y"], vocab,
        max_len=args.max_len, augment=False
    )
    test_ds = FalcoDataset(
        test_df["_sanitized_text"], test_df["_y"], vocab,
        max_len=args.max_len, augment=False
    )

    pin = torch.cuda.is_available()
    ssl_loader = DataLoader(
        ssl_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=pin, drop_last=True
    )
    cls_loader = DataLoader(
        cls_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=pin
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=pin
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[DEVICE] {device}")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    model = FalcoTransformer(
        vocab_size=len(vocab),
        max_len=args.max_len,
        d_model=args.d_model,
        nhead=args.heads,
        num_layers=args.layers,
        dim_ff=args.ff_dim,
        dropout=args.dropout,
        proj_dim=args.proj_dim
    ).to(device)

    # Phase 1: self-supervised contrastive pretraining
    ssl_opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr_ssl, weight_decay=args.weight_decay
    )
    train_ssl(
        model, ssl_loader, ssl_opt, device,
        epochs=args.epochs_ssl, temperature=args.temperature
    )

    # Phase 2: supervised fine-tuning on known benign/attack labels only
    cls_opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr_cls, weight_decay=args.weight_decay
    )
    train_classifier(model, cls_loader, cls_opt, device, epochs=args.epochs_cls)

    # Embeddings
    train_eval_loader = DataLoader(
        cls_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=pin
    )
    Xtr, Ltr, ytr = infer_embeddings(model, train_eval_loader, device)
    Xv, Lv, yv = infer_embeddings(model, val_loader, device)
    Xt, Lt, yt = infer_embeddings(model, test_loader, device)

    # Known-class classification quality
    pred_t = Lt.argmax(axis=1)
    p, r, f1, _ = precision_recall_fscore_support(
        yt, pred_t, average="binary", zero_division=0
    )
    acc = accuracy_score(yt, pred_t)

    # OOD
    ood = MahalanobisOOD(shrinkage=args.cov_shrinkage)
    ood.fit(Xtr, ytr)

    val_scores = ood.score(Xv)
    test_scores = ood.score(Xt)

    # Threshold estimated ONLY from known validation distribution.
    # e.g. 95th percentile => ~5% ID validation rejection.
    ood_threshold = float(np.quantile(val_scores, args.ood_quantile))

    y_ood = test_df["_is_ood"].to_numpy(dtype=int)
    pred_ood = (test_scores > ood_threshold).astype(int)

    ood_auroc = safe_auc(y_ood, test_scores)
    ood_aupr = safe_aupr(y_ood, test_scores)
    fpr95 = fpr_at_95_tpr(y_ood, test_scores)

    tp = int(((pred_ood == 1) & (y_ood == 1)).sum())
    fn = int(((pred_ood == 0) & (y_ood == 1)).sum())
    fp = int(((pred_ood == 1) & (y_ood == 0)).sum())
    tn = int(((pred_ood == 0) & (y_ood == 0)).sum())

    unknown_recall = tp / max(tp + fn, 1)
    unknown_precision = tp / max(tp + fp, 1)
    id_false_unknown_rate = fp / max(fp + tn, 1)

    # Final 3-state decision
    # 0=benign, 1=known_attack, 2=unknown_suspicious
    final_decision = np.where(
        pred_ood == 1,
        2,
        pred_t
    )

    leaked_train = int(((train_df[rule_col].astype(str) == str(heldout_rule)) & (train_df["_y"] == 1)).sum())
    leaked_val = int(((val_df[rule_col].astype(str) == str(heldout_rule)) & (val_df["_y"] == 1)).sum())
    print(f"[LEAK CHECK] heldout attacks in train={leaked_train} val={leaked_val}")
    if leaked_train or leaked_val:
        raise RuntimeError("Held-out attack leaked into train/validation")

    sweep_df = threshold_sweep(val_scores, test_scores, y_ood)

    result = {
        "heldout_rule": str(heldout_rule),
        "split_mode": split_mode,
        "heldout_leak_train": leaked_train,
        "heldout_leak_val": leaked_val,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "unseen_test_rows": int(y_ood.sum()),
        "classification_accuracy_all_test": float(acc),
        "classification_precision_attack_all_test": float(p),
        "classification_recall_attack_all_test": float(r),
        "classification_f1_attack_all_test": float(f1),
        "ood_threshold": ood_threshold,
        "ood_validation_quantile": float(args.ood_quantile),
        "ood_auroc": ood_auroc,
        "ood_aupr": ood_aupr,
        "fpr_at_95_tpr": fpr95,
        "unknown_precision": float(unknown_precision),
        "unknown_recall_detection_rate": float(unknown_recall),
        "id_false_unknown_rate": float(id_false_unknown_rate),
        "ood_confusion": {
            "tp_unknown": tp, "fp_unknown": fp,
            "tn_id": tn, "fn_unknown": fn
        }
    }

    print("\n--- RESULTS ---")
    for k, v in result.items():
        if k != "ood_confusion":
            print(f"{k}: {v}")
    print("ood_confusion:", result["ood_confusion"])
    print("\n--- OOD THRESHOLD SWEEP ---")
    print(sweep_df.to_string(index=False))

    # Save artifacts
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(heldout_rule))[:80]
    out_dir = Path(args.output_dir) / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(result, f, indent=2)

    sweep_df.to_csv(out_dir / "ood_threshold_sweep.csv", index=False)

    pred_df = test_df[[label_col, rule_col, text_col, "_sanitized_text", "_raw_pod"]].copy()
    pred_df["true_binary"] = yt
    pred_df["classifier_pred"] = pred_t
    pred_df["is_true_unseen"] = y_ood
    pred_df["ood_score"] = test_scores
    pred_df["ood_threshold"] = ood_threshold
    pred_df["pred_unknown"] = pred_ood
    pred_df["final_decision"] = final_decision
    pred_df["final_label"] = pred_df["final_decision"].map({
        0: "BENIGN",
        1: "KNOWN_ATTACK",
        2: "UNKNOWN_SUSPICIOUS"
    })
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    # Save model + vocab
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab_itos": vocab.itos,
            "args": vars(args),
            "heldout_rule": str(heldout_rule),
            "ood_means": ood.means,
            "ood_precision": ood.precision,
            "ood_threshold": ood_threshold
        },
        out_dir / "model_and_ood.pt"
    )

    print(f"[SAVED] {out_dir}")
    return result

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--csv", required=True, help="Falco/Kubernetes labeled CSV")
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--rule-col", default=None)
    ap.add_argument("--text-col", default=None)
    ap.add_argument("--timestamp-col", default=None)

    ap.add_argument("--heldout-rule", default=None,
                    help="Attack rule/family to remove completely from training")
    ap.add_argument("--auto-holdouts", type=int, default=0,
                    help="Automatically run top N attack rules as unseen")
    ap.add_argument("--list-rules", action="store_true",
                    help="Print attack rules and exit")
    ap.add_argument("--show-sanitization", type=int, default=0,
                    help="Print N original/sanitized examples and exit")

    ap.add_argument("--output-dir", default="k8s_falco_ood_results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-train-rows", type=int, default=0,
                    help="0 means use all training rows")

    ap.add_argument("--vocab-size", type=int, default=30000)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--ff-dim", type=int, default=256)
    ap.add_argument("--proj-dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--token-drop", type=float, default=0.12)

    ap.add_argument("--epochs-ssl", type=int, default=5)
    ap.add_argument("--epochs-cls", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr-ssl", type=float, default=2e-4)
    ap.add_argument("--lr-cls", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.15)

    ap.add_argument("--cov-shrinkage", type=float, default=1e-3)
    ap.add_argument("--ood-quantile", type=float, default=0.95,
                    help="Known-validation quantile used as OOD threshold")

    args = ap.parse_args()
    if args.max_train_rows == 0:
        args.max_train_rows = None

    seed_everything(args.seed)

    print(f"[LOAD] {args.csv}")
    df = pd.read_csv(args.csv, low_memory=False)
    print(f"[DATA] shape={df.shape}")

    label_col = infer_col(df, "label", args.label_col, required=True)
    rule_col = infer_col(df, "rule", args.rule_col, required=True)
    text_col = infer_col(df, "text", args.text_col, required=True)
    timestamp_col = infer_col(df, "timestamp", args.timestamp_col, required=False)

    print(f"[COLUMNS] label={label_col} rule={rule_col} text={text_col} timestamp={timestamp_col}")

    # Remove unusable rows
    df = df.dropna(subset=[label_col, rule_col, text_col]).copy()
    df["_y"] = normalize_binary_label(df[label_col])

    print("[INFO] extracting raw pod groups and sanitizing Falco text...")
    df["_raw_pod"] = df[text_col].astype(str).map(extract_raw_pod_name)
    df["_group_id"] = build_group_ids(df, text_col)
    df["_sanitized_text"] = df[text_col].astype(str).map(sanitize_falco_text)

    if timestamp_col is not None:
        try:
            df["_ts"] = pd.to_datetime(df[timestamp_col], errors="coerce")
            df = df.sort_values("_ts", kind="stable").reset_index(drop=True)
            print("[INFO] data ordered by timestamp")
        except Exception:
            pass

    print(f"[LABELS] benign={(df['_y']==0).sum():,} attack={(df['_y']==1).sum():,}")
    attack_counts = (
        df[df["_y"] == 1]
        .groupby(rule_col)
        .size()
        .sort_values(ascending=False)
    )

    if args.show_sanitization > 0:
        n = int(args.show_sanitization)
        print("\nSANITIZATION EXAMPLES:")
        print(df[[rule_col, label_col, text_col, "_sanitized_text", "_raw_pod"]].head(n).to_string(index=False))
        return

    if args.list_rules:
        print("\nATTACK RULES:")
        print(attack_counts.to_string())
        return

    if args.heldout_rule:
        holdouts = [args.heldout_rule]
    elif args.auto_holdouts > 0:
        holdouts = top_attack_rules(
            df, rule_col, n=args.auto_holdouts, min_rows=50
        )
        if not holdouts:
            raise RuntimeError("No attack rules with at least 50 rows found.")
        print(f"[AUTO HOLDOUTS] {holdouts}")
    else:
        raise ValueError(
            "Choose one of:\n"
            "  --heldout-rule 'RULE NAME'\n"
            "  --auto-holdouts 5\n"
            "  --list-rules"
        )

    all_results = []
    for rule in holdouts:
        seed_everything(args.seed)
        result = run_one(
            args, df, label_col, rule_col, text_col, rule
        )
        all_results.append(result)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_results).to_csv(
        out_root / "all_holdout_summary.csv", index=False
    )
    with open(out_root / "all_holdout_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*90)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*90)
    print(pd.DataFrame(all_results)[[
        "heldout_rule",
        "unseen_test_rows",
        "ood_auroc",
        "ood_aupr",
        "unknown_recall_detection_rate",
        "id_false_unknown_rate"
    ]].to_string(index=False))
    print(f"\n[SAVED] {out_root / 'all_holdout_summary.csv'}")

if __name__ == "__main__":
    main()
