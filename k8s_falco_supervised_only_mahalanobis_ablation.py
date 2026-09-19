#!/usr/bin/env python3
"""
Supervised-only Transformer + Mahalanobis OOD ablation.

This wrapper reuses the FINAL k8s_falco_contrastive_ood_v2.py implementation
for:
  - preprocessing/sanitization
  - pod/group-aware splitting
  - rule-associated malicious holdout
  - vocabulary construction
  - Transformer architecture
  - supervised classifier training
  - CLS embedding extraction
  - Mahalanobis OOD scoring
  - validation-derived thresholds
  - all OOD metrics and artifact saving

The ONLY methodological change is that self-supervised contrastive
pretraining is disabled. The Transformer starts from random initialization
and is trained only with the supervised benign/attack objective.

This is intentionally a wrapper rather than a copied/reimplemented pipeline
so the ablation remains aligned with the final baseline implementation.
"""

import sys
import argparse

import k8s_falco_contrastive_ood_v2 as base


def _skip_contrastive_pretraining(*args, **kwargs):
    """Disable Phase 1 while preserving the rest of the final pipeline."""
    print("[ABLATION] Contrastive pretraining DISABLED.")
    print("[ABLATION] Training starts from random Transformer initialization.")
    print("[ABLATION] Proceeding directly to supervised benign/attack fine-tuning.")


def main():
    # Replace the original train_ssl function with a no-op.
    base.train_ssl = _skip_contrastive_pretraining

    ap = argparse.ArgumentParser(
        description=(
            "Supervised-only Transformer + Mahalanobis OOD ablation "
            "using the final Kubernetes/Falco V2 pipeline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument(
        "--csv",
        default="/data/PyCharmMiscProject/Kubernates/clean_falco_alerts.csv",
        help="Falco/Kubernetes labeled CSV",
    )
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--rule-col", default="rule_name")
    ap.add_argument("--text-col", default="alert_text")
    ap.add_argument("--timestamp-col", default="timestamp")

    ap.add_argument(
        "--heldout-rule",
        default=None,
        help="Run one specific rule-associated malicious holdout.",
    )
    ap.add_argument(
        "--auto-holdouts",
        type=int,
        default=3,
        help=(
            "Number of highest-count attack rules with >=50 malicious events "
            "to evaluate. Default 3 = the three primary holdouts."
        ),
    )

    ap.add_argument(
        "--output-dir",
        default="/data/PyCharmMiscProject/Kubernates/"
                "k8s_falco_ood_results_supervised_only_mahalanobis",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-train-rows", type=int, default=0)

    # Same model/data settings as the final V2 implementation.
    ap.add_argument("--vocab-size", type=int, default=30000)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--ff-dim", type=int, default=256)
    ap.add_argument("--proj-dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.10)

    # Retained for CLI compatibility; not used because SSL is disabled.
    ap.add_argument("--token-drop", type=float, default=0.12)
    ap.add_argument("--epochs-ssl", type=int, default=0)
    ap.add_argument("--epochs-cls", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr-ssl", type=float, default=2e-4)
    ap.add_argument("--lr-cls", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.15)

    ap.add_argument("--cov-shrinkage", type=float, default=1e-3)
    ap.add_argument("--ood-quantile", type=float, default=0.95)

    args = ap.parse_args()

    # Build argv exactly as expected by the final V2 main().
    forwarded = [
        "k8s_falco_contrastive_ood_v2.py",
        "--csv", args.csv,
        "--label-col", args.label_col,
        "--rule-col", args.rule_col,
        "--text-col", args.text_col,
        "--timestamp-col", args.timestamp_col,
        "--output-dir", args.output_dir,
        "--seed", str(args.seed),
        "--workers", str(args.workers),
        "--max-train-rows", str(args.max_train_rows),
        "--vocab-size", str(args.vocab_size),
        "--min-freq", str(args.min_freq),
        "--max-len", str(args.max_len),
        "--d-model", str(args.d_model),
        "--heads", str(args.heads),
        "--layers", str(args.layers),
        "--ff-dim", str(args.ff_dim),
        "--proj-dim", str(args.proj_dim),
        "--dropout", str(args.dropout),
        "--token-drop", str(args.token_drop),
        "--epochs-ssl", "0",
        "--epochs-cls", str(args.epochs_cls),
        "--batch-size", str(args.batch_size),
        "--lr-ssl", str(args.lr_ssl),
        "--lr-cls", str(args.lr_cls),
        "--weight-decay", str(args.weight_decay),
        "--temperature", str(args.temperature),
        "--cov-shrinkage", str(args.cov_shrinkage),
        "--ood-quantile", str(args.ood_quantile),
    ]

    if args.heldout_rule:
        forwarded += ["--heldout-rule", args.heldout_rule]
    else:
        forwarded += ["--auto-holdouts", str(args.auto_holdouts)]

    if args.cpu:
        forwarded.append("--cpu")

    print("=" * 90)
    print("SUPERVISED-ONLY TRANSFORMER + MAHALANOBIS OOD ABLATION")
    print("=" * 90)
    print(f"[CSV] {args.csv}")
    print(f"[OUTPUT] {args.output_dir}")
    print("[CHANGE] Contrastive pretraining removed; all other V2 components retained.")
    print("[MODEL] 128-D CLS, 4 heads, 2 layers, FF=256, dropout=0.10")
    print("[TRAIN] supervised epochs=5, batch=256, lr=1e-4, wd=1e-4")
    print("[OOD] class-conditional Mahalanobis, covariance regularization=1e-3")
    print("[THRESHOLD] validation q=0.95; V2 sweep is also retained")
    print("[HOLDOUTS] primary rules only by default (top 3 with >=50 attacks)")
    print("=" * 90)

    old_argv = sys.argv
    try:
        sys.argv = forwarded
        base.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
