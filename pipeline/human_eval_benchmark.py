import json
import argparse
import os

def generate_report(output_path, problems):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Human Evaluation Benchmark\n\n")
        f.write("Review the generated code for the following problems.\n\n")
        for i, problem in enumerate(problems, 1):
            f.write(f"## Problem {i}\n")
            f.write(f"**Prompt:**\n```text\n{problem['prompt']}\n```\n\n")
            f.write(f"**Generated Code:**\n```python\n{problem['generated_code']}\n```\n\n")
            f.write("### Rubric\n")
            f.write("- [ ] **Correctness:** Does the code solve the problem accurately?\n")
            f.write("- [ ] **Readability:** Is the code clean, well-structured, and idiomatic?\n")
            f.write("- [ ] **Efficiency:** Is the approach optimal in terms of time and space complexity?\n")
            f.write("- [ ] **Documentation:** Are there appropriate docstrings, type hints, and comments?\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate human evaluation benchmark report")
    parser.add_argument("--output", type=str, default="human_eval_report.md", help="Output markdown file path")
    args = parser.parse_args()
    
    # Example samples - in a real scenario these would be loaded from a model generation output JSON
    samples = [
        {
            "prompt": "Write a Python function to compute the Fibonacci sequence up to n.",
            "generated_code": "def fibonacci(n: int) -> list[int]:\n    \"\"\"Compute Fibonacci sequence up to n elements.\"\"\"\n    if n <= 0: return []\n    if n == 1: return [0]\n    res = [0, 1]\n    while len(res) < n:\n        res.append(res[-1] + res[-2])\n    return res"
        },
        {
            "prompt": "Create a function to check if a string is a palindrome.",
            "generated_code": "def is_palindrome(s: str) -> bool:\n    \"\"\"Check if string is a palindrome.\"\"\"\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]"
        },
        {
            "prompt": "Write a function that returns the prime factors of a given number.",
            "generated_code": "def prime_factors(n: int) -> list[int]:\n    \"\"\"Return the prime factors of n.\"\"\"\n    i = 2\n    factors = []\n    while i * i <= n:\n        if n % i:\n            i += 1\n        else:\n            n //= i\n            factors.append(i)\n    if n > 1:\n        factors.append(n)\n    return factors"
        }
    ]
    
    generate_report(args.output, samples)
    print(f"Human evaluation report generated at {args.output}")
