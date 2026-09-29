"""Training for the stride-2 point detector (PLAN.md Phase 5).

``data`` is numpy only; ``model``, ``losses``, ``train`` and ``export`` need
PyTorch and timm, which only training machines (Kaggle, or a CPU smoke run)
install. Inference reads the exported ONNX model through core.inference.
"""
