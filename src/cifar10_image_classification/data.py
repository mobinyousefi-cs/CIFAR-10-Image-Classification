#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: CIFAR-10 Image Classification with Keras
File: data.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-08
Updated: 2025-11-08
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Data loading and preprocessing utilities for the CIFAR-10 dataset.

Usage:
from cifar10_image_classification.config import TrainingConfig
from cifar10_image_classification.data import create_datasets

config = TrainingConfig()
train_ds, val_ds, test_ds = create_datasets(config)

Notes:
- Uses tf.data for efficient input pipelines.
- Normalization to [0, 1] is handled here; the model adds any extra preprocessing.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf

from .config import TrainingConfig


def _normalize_images(x: np.ndarray) -> np.ndarray:
    """Normalize image pixels to the [0, 1] range as float32."""

    x = x.astype("float32") / 255.0
    return x


def _split_train_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    validation_split: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the original training set into train and validation subsets."""

    num_train = x_train.shape[0]
    num_val = int(num_train * validation_split)

    indices = np.arange(num_train)
    np.random.shuffle(indices)

    val_indices = indices[:num_val]
    train_indices = indices[num_val:]

    x_val = x_train[val_indices]
    y_val = y_train[val_indices]
    x_train_new = x_train[train_indices]
    y_train_new = y_train[train_indices]

    return x_train_new, y_train_new, x_val, y_val


def _build_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
) -> tf.data.Dataset:
    """Create a tf.data.Dataset from numpy arrays."""

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(x))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def create_datasets(config: TrainingConfig) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load CIFAR-10 and create train/val/test datasets.

    Parameters
    ----------
    config:
        Training configuration with batch size, validation split, etc.

    Returns
    -------
    train_ds, val_ds, test_ds : tf.data.Dataset
        Datasets ready to be consumed by a Keras model.
    """

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # y_* are of shape (N, 1); we prefer 1D for sparse labels.
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    x_train = _normalize_images(x_train)
    x_test = _normalize_images(x_test)

    x_train, y_train, x_val, y_val = _split_train_validation(
        x_train=x_train,
        y_train=y_train,
        validation_split=config.validation_split,
    )

    train_ds = _build_dataset(x_train, y_train, batch_size=config.batch_size, shuffle=True)
    val_ds = _build_dataset(x_val, y_val, batch_size=config.batch_size, shuffle=False)
    test_ds = _build_dataset(x_test, y_test, batch_size=config.batch_size, shuffle=False)

    return train_ds, val_ds, test_ds
