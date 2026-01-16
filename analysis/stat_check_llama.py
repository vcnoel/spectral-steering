import json
import numpy as np
import math
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint
import os

INPUT_FILE = "output/llama_rigorous_results.json"
CONDITIONS = ["baseline", "control_random", "spectral_steer"] 

def calculate_cohens_h(p1, p2):
    # h = 2 * (asin(sqrt(p1)) - asin(sqrt(p2)))
    # p1 = baseline, p2 = steered
    p1 = max(1e-6, min(1-1e-6, p1))
    p2 = max(1e-6, min(1-1e-6, p2))
    return 2 * (math.asin(math.sqrt(p2)) - math.asin(math.sqrt(p1)))

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run experiments/19_llama_rigorous_validation.py first.")
        return

    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    # N is 300
    N = 300
    
    print(f"\n--- Llama 3.2 3B Statistical Report (N={N}) ---\n")
    
    baseline_acc = data["baseline"]
    
    h_header = "Cohen's h"
    print(f"{'Condition':<20} | {'Acc':<8} | {'95% CI (Wilson)':<18} | {h_header:<10}")
    print("-" * 75)

    for cond in CONDITIONS:
        acc = data.get(cond, 0.0)
        correct_count = int(acc * N)
        
        # 1. Agresti-Coull / Wilson CI
        ci_low, ci_high = proportion_confint(correct_count, N, method='wilson')
        ci_str = f"[{ci_low:.1%}, {ci_high:.1%}]"
        
        # 2. Cohen's h
        h_str = "——"
        if cond != "baseline":
            h_val = calculate_cohens_h(baseline_acc, acc)
            h_str = f"{h_val:.2f}"
            
        print(f"{cond:<20} | {acc:.1%}    | {ci_str:<18} | {h_str:<10}")

    print("-" * 75)
    print("Interpretation:")
    print("Cohen's h > 0.5 is a LARGE effect.")
    print("Llama 3.2 3B is expected to show the 'Void' pattern at Layer 2.")

if __name__ == "__main__":
    main()
