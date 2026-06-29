import pytest
import sys
import os

# Add the pipeline directory to the python path so we can import it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_prep import is_valid

def test_is_valid_accepts_good_data():
    example = {
        "text": "small text",
        "output": "def my_func():\n    return 42" + " " * 30, # ensure > 30 len
        "instruction": "Write a python function"
    }
    assert is_valid(example, max_chars=100) is True

def test_is_valid_rejects_long_text():
    example = {
        "text": "a" * 150,
        "output": "def my_func():\n    return 42" + " " * 30,
        "instruction": "Write a python function"
    }
    assert is_valid(example, max_chars=100) is False

def test_is_valid_rejects_missing_keywords():
    example = {
        "text": "small text",
        "output": "This is just english text with no python code." + " " * 30,
        "instruction": "Write a python function"
    }
    # It doesn't contain "def ", "class ", "import ", or "="
    assert is_valid(example, max_chars=100) is False

def test_is_valid_operator_precedence():
    # Before the fix, if output contained "=", it would return True regardless of max_chars.
    example = {
        "text": "a" * 150,  # too long
        "output": "x = 5" + " " * 30,
        "instruction": "Write a python function"
    }
    # Because of max_chars, it MUST be False. 
    # If the operator precedence bug exists, it would be True.
    assert is_valid(example, max_chars=100) is False
