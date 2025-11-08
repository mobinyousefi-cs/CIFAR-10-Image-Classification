#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: CIFAR-10 Image Classification with Keras
File: model.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-08
Updated: 2025-11-08
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Model definition and compilation utilities for the CIFAR-10 CNN.

Usage:
from cifar10_image_classification.config import TrainingConfig
from cifar10_image_classification.model import build_compiled_model

config = TrainingConfig()
model = build_compiled_model(config)

Notes:
- The architecture is intentionally moderate-sized for educational purposes.
- You can easily extend this with more layers or modern blocks (e.g., ResNet).
"""

from __future__ import annotations

import tensorflow as tf

from .config import TrainingConfig


def build_cifar10_cnn(config: TrainingConfig) -> tf.keras.Model:
    """Build a simple but effective CNN for CIFAR-10."""

    inputs = tf.keras.Input(
        shape=(config.image_height, config.image_width, config.num_channels), name="images"
    )

    # Basic preprocessing
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)

    # Optional data augmentation (only active during training)
    x = tf.keras.layers.RandomFlip("horizontal", name="rand_flip")(x)
    x = tf.keras.layers.RandomRotation(0.1, name="rand_rot")(x)

    # Conv block 1
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation=None, name="conv1_1")(x)
    x = tf.keras.layers.BatchNormalization(name="bn1_1")(x)
    x = tf.keras.layers.Activation("relu", name="relu1_1")(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation=None, name="conv1_2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn1_2")(x)
    x = tf.keras.layers.Activation("relu", name="relu1_2")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = tf.keras.layers.Dropout(0.25, name="dropout1")(x)

    # Conv block 2
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=None, name="conv2_1")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2_1")(x)
    x = tf.keras.layers.Activation("relu", name="relu2_1")(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=None, name="conv2_2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2_2")(x)
    x = tf.keras.layers.Activation("relu", name="relu2_2")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = tf.keras.layers.Dropout(0.35, name="dropout2")(x)

    # Conv block 3
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation=None, name="conv3_1")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3_1")(x)
    x = tf.keras.layers.Activation("relu", name="relu3_1")(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation=None, name="conv3_2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3_2")(x)
    x = tf.keras.layers.Activation("relu", name="relu3_2")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout3")(x)

    # Classifier head
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(256, activation=None, name="dense1")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_dense1")(x)
    x = tf.keras.layers.Activation("relu", name="relu_dense1")(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout_dense1")(x)

    outputs = tf.keras.layers.Dense(config.num_classes, activation="softmax", name="logits")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cifar10_cnn")
    return model


def build_compiled_model(config: TrainingConfig) -> tf.keras.Model:
    """Build and compile the CIFAR-10 CNN model."""

    model = build_cifar10_cnn(config)

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model
