#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: CIFAR-10 Image Classification with Keras
File: test_model.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-08
Updated: 2025-11-08
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Basic tests for the CIFAR-10 model construction and forward pass.

Usage:
pytest tests/test_model.py

Notes:
- Uses a small dummy batch for a quick forward pass.
"""

from __future__ import annotations

import numpy as np

from cifar10_image_classification.config import TrainingConfig
from cifar10_image_classification.model import build_compiled_model


def test_build_compiled_model_forward_pass() -> None:
    config = TrainingConfig(batch_size=8)
    model = build_compiled_model(config)

    dummy_x = np.random.rand(8, config.image_height, config.image_width, config.num_channels).astype(
        "float32"
    )
    outputs = model.predict(dummy_x, verbose=0)

    assert outputs.shape == (8, config.num_classes)
