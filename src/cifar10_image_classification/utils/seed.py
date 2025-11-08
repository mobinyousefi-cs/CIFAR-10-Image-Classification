#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: CIFAR-10 Image Classification with Keras
File: seed.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-08
Updated: 2025-11-08
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Utility functions to improve reproducibility by setting global random seeds.

Usage:
from cifar10_image_classification.utils.seed import set_global_determinism
set_global_determinism(42)

Notes:
- Full determinism is not guaranteed on all hardware / driver combinations,
  but this helps to reduce run-to-run variance.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import tensorflow as tf


def set_global_determinism(seed: int = 42, *, deterministic_ops: bool = True) -> None:
    """Set seeds for Python, NumPy, and TensorFlow.

    Parameters
    ----------
    seed:
        Seed value to use for all RNGs.
    deterministic_ops:
        If True, request deterministic ops from TensorFlow when possible.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if deterministic_ops:
        # Best-effort deterministic behavior for TF 2.x
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
        os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")


def get_seed_from_env(default: int = 42, env_var: str = "CIFAR10_SEED") -> int:
    """Read a seed value from environment variables, or fall back to default."""

    value: Optional[str] = os.getenv(env_var)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default
