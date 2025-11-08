#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: CIFAR-10 Image Classification with Keras
File: train.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-08
Updated: 2025-11-08
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Training script for the CIFAR-10 CNN model.

Usage:
python -m cifar10_image_classification.train \
    --epochs 30 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --model-dir artifacts/models

Notes:
- Uses EarlyStopping and ModelCheckpoint callbacks.
- Metrics are also logged for TensorBoard visualization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from .config import TrainingConfig
from .data import create_datasets
from .model import build_compiled_model
from .utils.logger import get_logger
from .utils.seed import get_seed_from_env, set_global_determinism


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""

    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="artifacts/models",
        help="Directory to store trained models.",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point for model training."""

    args = parse_args()

    seed = get_seed_from_env(default=42)
    set_global_determinism(seed)
    logger.info("Using random seed: %d", seed)

    config = TrainingConfig(
        seed=seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model_dir=Path(args.model_dir),
    )
    config.ensure_directories()

    logger.info("Loading CIFAR-10 dataset …")
    train_ds, val_ds, test_ds = create_datasets(config)

    logger.info("Building and compiling the model …")
    model = build_compiled_model(config)
    model.summary(print_fn=lambda x: logger.info("%s", x))

    best_model_path = config.model_dir / "best_model.h5"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(config.log_dir),
            histogram_freq=1,
        ),
    ]

    logger.info("Starting training for %d epochs …", config.epochs)
    history = model.fit(
        train_ds,
        epochs=config.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
    )

    # Save final model as well
    final_model_path = config.model_dir / "final_model.h5"
    model.save(final_model_path)
    logger.info("Training completed. Best model saved to: %s", best_model_path)
    logger.info("Final model saved to: %s", final_model_path)

    logger.info("Evaluating best model on the test set …")
    best_model = tf.keras.models.load_model(best_model_path)
    test_loss, test_acc = best_model.evaluate(test_ds, verbose=0)
    logger.info("Test loss: %.4f | Test accuracy: %.4f", test_loss, test_acc)


if __name__ == "__main__":  # pragma: no cover
    main()
