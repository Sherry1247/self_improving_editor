"""Detector module exports."""

from .base import Detector
from .yolo_detector import YOLODetector

__all__ = ["Detector", "YOLODetector"]
