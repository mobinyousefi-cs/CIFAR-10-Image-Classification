#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: CIFAR-10 Image Classification with Keras
File: config.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-08
Updated: 2025-11-08
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Central configuration objects and constants for the CIFAR-10 image classification project.

Usage:
from cifar10_image_classification.config import TrainingConfig, CLASS_NAMES

Notes:
- Adjust hyperparameters here to experiment with different training regimes.
- Directories for models and logs are created automatically if they do not exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

PROJECT_NAME: str = "CIFAR-10 Image Classification with Keras"

CLASS_NAMES: List[str] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


@dataclass
class TrainingConfig:
    """Configuration options for training and evaluation."""

    seed: int = 42
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 1e-3
    num_classes: int = 10

    image_height: int = 32
    image_width: int = 32
    num_channels: int = 3

    validation_split: float = 0.1

    model_dir: Path = Path("artifacts") / "models"
    log_dir: Path = Path("artifacts") / "logs"

    def ensure_directories(self) -> None:
        """Create required directories if they do not exist."""

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
