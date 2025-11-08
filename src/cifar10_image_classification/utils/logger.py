#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: CIFAR-10 Image Classification with Keras
File: logger.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-08
Updated: 2025-11-08
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Central logging configuration for the project.

Usage:
from cifar10_image_classification.utils.logger import get_logger
logger = get_logger(__name__)
logger.info("Training started")

Notes:
- Logging format is kept concise but informative.
"""

from __future__ import annotations

import logging
from typing import Optional


_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _configure_root_logger(level: int = logging.INFO) -> None:
    """Configure the root logger if it has no handlers yet."""

    root = logging.getLogger()
    if root.handlers:
        return

    handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: Optional[str] = None, *, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a standard configuration.

    Parameters
    ----------
    name:
        Name of the logger; if None, the root logger is returned.
    level:
        Logging level (e.g., logging.INFO, logging.DEBUG).
    """

    _configure_root_logger(level=level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
