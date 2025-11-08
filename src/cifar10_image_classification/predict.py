#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: CIFAR-10 Image Classification with Keras
File: predict.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-08
Updated: 2025-11-08
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Command-line script for predicting the class of a single image using a trained CIFAR-10 model.

Usage:
python -m cifar10_image_classification.predict \
    --model-path artifacts/models/best_model.h5 \
    --image-path path/to/image.png

Notes:
- The image is resized to 32x32 and normalized to [0, 1].
- The predicted class name and probabilities are printed to stdout.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

from .config import CLASS_NAMES, TrainingConfig
from .utils.logger import get_logger
from .utils.seed import get_seed_from_env, set_global_determinism


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for prediction."""

    parser = argparse.ArgumentParser(
        description="Predict the CIFAR-10 class for a single input image."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="artifacts/models/best_model.h5",
        help="Path to the trained model file.",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="Path to the input image file.",
    )
    return parser.parse_args()


def _load_and_preprocess_image(
    image_path: Path,
    target_size: Tuple[int, int] = (32, 32),
) -> np.ndarray:
    """Load an image from disk and preprocess it for the model."""

    img = tf.keras.utils.load_img(image_path, target_size=target_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, C)
    return img_array


def main() -> None:
    """Entry point for single-image prediction."""

    args = parse_args()

    seed = get_seed_from_env(default=42)
    set_global_determinism(seed)
    logger.info("Using random seed: %d", seed)

    _ = TrainingConfig(seed=seed)  # Reserved for future use if needed

    model_path = Path(args.model_path)
    image_path = Path(args.image_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    logger.info("Loading model from: %s", model_path)
    model = tf.keras.models.load_model(model_path)

    logger.info("Loading and preprocessing image: %s", image_path)
    img_array = _load_and_preprocess_image(image_path)

    logger.info("Running inference …")
    probabilities = model.predict(img_array, verbose=0)[0]  # shape: (num_classes,)

    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])

    logger.info("Predicted class: %s (index=%d, confidence=%.4f)", predicted_class, predicted_index, confidence)

    # Pretty-print class probabilities
    logger.info("Class probabilities:")
    for idx, (cls_name, prob) in enumerate(zip(CLASS_NAMES, probabilities)):
        logger.info("  [%d] %-10s : %.4f", idx, cls_name, float(prob))


if __name__ == "__main__":  # pragma: no cover
    main()
