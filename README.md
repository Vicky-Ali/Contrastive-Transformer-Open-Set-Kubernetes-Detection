# Contrastive Transformer-Based Open-Set Intrusion Detection for Kubernetes Using Falco Runtime Alerts

This repository contains the implementation accompanying the research study on contrastive Transformer-based open-set intrusion detection for Kubernetes environments using Falco runtime alerts.

The proposed pipeline learns behavioral representations from Falco security alerts and combines supervised classification with Mahalanobis-distance-based out-of-distribution (OOD) detection to identify previously unseen attack behaviors.

## Overview

Traditional intrusion detection systems are commonly evaluated on attack classes or patterns observed during training. In Kubernetes environments, however, previously unseen attack behaviors may emerge after deployment.

This work investigates an open-set detection approach in which:

1. Falco runtime alerts are preprocessed and sanitized.
2. A Transformer encoder learns representations of alert behavior.
3. Self-supervised contrastive pretraining is used to improve behavioral representations.
4. The learned representations are fine-tuned using supervised benign/attack classification.
5. CLS embeddings are extracted from the trained Transformer.
6. Class-conditional Mahalanobis distances are used for OOD detection.
7. Validation data are used to determine OOD decision thresholds.
8. Held-out malicious rules are used to evaluate detection of previously unseen attack behavior.

## Repository Contents

```text
.
├── k8sFalcoContrastiveOOD.py
└── k8sFalcoSupervisedOnlyMahalanobis.py
