#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: CIFAR-10 Image Classification with Keras
File: evaluate.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-08
Updated: 2025-11-08
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Evaluate a trained CIFAR-10 model on the test set.

Usage:
python -m cifar10_image_classification.evaluate \
    --model-path artifacts/models/best_model.h5

Notes:
- Assumes the model was trained using `train.py` and saved in HDF5 format.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from .config import TrainingConfig
from .data import create_datasets
from .utils.logger import get_logger
from .utils.seed import get_seed_from_env, set_global_determinism


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate a CIFAR-10 model on the test set.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="artifacts/models/best_model.h5",
        help="Path to the trained model file (HDF5 or SavedModel).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for model evaluation."""

    args = parse_args()

    seed = get_seed_from_env(default=42)
    set_global_determinism(seed)
    logger.info("Using random seed: %d", seed)

    config = TrainingConfig(seed=seed)

    logger.info("Loading CIFAR-10 dataset …")
    _, _, test_ds = create_datasets(config)

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info("Loading model from: %s", model_path)
    model = tf.keras.models.load_model(model_path)

    logger.info("Evaluating on the test set …")
    results = model.evaluate(test_ds, verbose=1)
    metrics = dict(zip(model.metrics_names, results))

    logger.info("Test metrics:")
    for name, value in metrics.items():
        logger.info("  %s: %.4f", name, value)


if __name__ == "__main__":  # pragma: no cover
    main()
