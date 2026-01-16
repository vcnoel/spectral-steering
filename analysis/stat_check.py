import json
import numpy as np
import math
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests

INPUT_FILE = "output/rigorous_results_N300.json"
CONDITIONS = ["Baseline", "Control_Random", "Control_Anti", "Spectral"]

def calculate_cohens_h(p1, p2):
    # h = 2 * (asin(sqrt(p1)) - asin(sqrt(p2)))
    # p1 = baseline, p2 = steered
    return 2 * (math.asin(math.sqrt(p2)) - math.asin(math.sqrt(p1)))

def main():
    try:
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Run experiments/15_rigorous_validation.py first.")
        return

    N = data["N"]
    print(f"\n--- Statistical Report (N={N}) ---\n")
    
    baseline_results = data["Baseline"]["results"]
    baseline_acc = data["Baseline"]["accuracy"]
    
    print(f"{'Condition':<20} | {'Acc':<8} | {'95% CI (Wilson)':<18} | {'Cohen\'s h':<10} | {'McNemar p':<12}")
    print("-" * 85)

    # Process each condition
    for cond in CONDITIONS:
        cond_data = data.get(cond)
        if not cond_data: continue
        
        results = cond_data["results"] # Dict of id -> bool
        acc = cond_data["accuracy"]
        correct_count = int(acc * N)
        
        # 1. Agresti-Coull / Wilson CI
        ci_low, ci_high = proportion_confint(correct_count, N, method='wilson')
        ci_str = f"[{ci_low:.1%}, {ci_high:.1%}]"
        
        # 2. Comparison to Baseline (if not baseline itself)
        h_val = 0.0
        p_val = 1.0
        
        if cond != "Baseline":
            # Cohen's h
            h_val = calculate_cohens_h(baseline_acc, acc)
            
            # McNemar's Test
            # Construct Contingency Table
            #               Cond Fail   Cond Pass
            # Base Fail        [00]       [01]
            # Base Pass        [10]       [11]
            
            both_fail = 0
            base_fail_cond_pass = 0
            base_pass_cond_fail = 0
            both_pass = 0
            
            for pid, base_res in baseline_results.items():
                cond_res = results.get(pid, False)
                
                if base_res and cond_res: both_pass += 1
                elif not base_res and not cond_res: both_fail += 1
                elif not base_res and cond_res: base_fail_cond_pass += 1 # Improvement
                elif base_res and not cond_res: base_pass_cond_fail += 1 # Regression
            
            table = [[both_fail, base_fail_cond_pass],
                     [base_pass_cond_fail, both_pass]]
            
            # Exact McNemar
            # If zeros in off-diagonal, p-value can be 1.0 or small. 
            if base_fail_cond_pass + base_pass_cond_fail > 0:
                mcnemar_res = mcnemar(table, exact=True)
                p_val = mcnemar_res.pvalue
            else:
                p_val = 1.0

            p_str = f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.2f}"
            h_str = f"{h_val:.2f}"
        else:
            p_str = "——"
            h_str = "——"

        print(f"{cond:<20} | {acc:.1%}    | {ci_str:<18} | {h_str:<10} | {p_str:<12}")

    print("-" * 85)
    print("Interpretation:")
    print("Cohen's h > 0.5 is a LARGE effect.")
    print("McNemar p < 0.05 indicates statistically significant difference from Baseline.")

if __name__ == "__main__":
    main()
