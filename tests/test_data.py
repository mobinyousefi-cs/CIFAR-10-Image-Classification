#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: CIFAR-10 Image Classification with Keras
File: test_data.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-08
Updated: 2025-11-08
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Basic tests for the CIFAR-10 data pipeline.

Usage:
pytest tests/test_data.py

Notes:
- These tests are lightweight sanity checks, not exhaustive validations.
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf

from cifar10_image_classification.config import TrainingConfig
from cifar10_image_classification.data import create_datasets


def _take_one_batch(ds: tf.data.Dataset) -> Tuple[tf.Tensor, tf.Tensor]:
    return next(iter(ds))


def test_create_datasets_shapes() -> None:
    config = TrainingConfig(batch_size=32)
    train_ds, val_ds, test_ds = create_datasets(config)

    x_batch, y_batch = _take_one_batch(train_ds)

    assert x_batch.ndim == 4  # (B, H, W, C)
    assert y_batch.ndim == 1  # (B,)

    assert x_batch.shape[1:] == (
        config.image_height,
        config.image_width,
        config.num_channels,
    )

    # Just check that validation and test datasets are non-empty
    x_val, y_val = _take_one_batch(val_ds)
    x_test, y_test = _take_one_batch(test_ds)

    assert x_val.shape[0] > 0
    assert x_test.shape[0] > 0
    assert y_val.shape[0] > 0
    assert y_test.shape[0] > 0
