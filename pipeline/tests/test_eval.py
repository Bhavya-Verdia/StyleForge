import pytest
import sys
import os
from unittest.mock import MagicMock

sys.modules['transformers'] = MagicMock()
sys.modules['peft'] = MagicMock()
sys.modules['torch'] = MagicMock()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eval import compute_rouge

def test_compute_rouge():
    preds = ["def add(a, b): return a + b"]
    refs = ["def add(a, b): return a + b"]
    scores = compute_rouge(preds, refs)
    
    assert "rouge1" in scores
    assert "rouge2" in scores
    assert "rougeL" in scores
    assert scores["rouge1"] == 1.0  # exact match

def test_compute_rouge_partial():
    preds = ["def add(a, b): return a + c"]
    refs = ["def add(a, b): return a + b"]
    scores = compute_rouge(preds, refs)
    
    assert scores["rouge1"] < 1.0
    assert scores["rouge1"] > 0.0
