import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from benchmarks import run_all

parser = argparse.ArgumentParser(description="Debug dataset metric calculations")
parser.add_argument("text", nargs="*", help="Text to analyze")
args = parser.parse_args()

if not args.text:
    print("Provide some text, e.g.:\n  python debug_dataset_metrics.py 'This is a sentence.'")
else:
    sample = " ".join(args.text)
    metrics = run_all(sample)
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

