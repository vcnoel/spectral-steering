import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import glob

def main():
    # Gather Data
    data_points = []
    
    # 1. Dosage Sweep (L12)
    try:
        with open('output/llama_dosage_sweep.json', 'r') as f:
            d = json.load(f)
            # L12 Base
            data_points.append({"Layer": 12, "Condition": "Baseline", "Accuracy": d.get("Baseline", 0)})
            # L12 Best Micro-Dose
            data_points.append({"Layer": 12, "Condition": "Micro-Sharpen", "Accuracy": d.get("L12_Strength_-0.01", 0)})
    except: pass
    
    # 2. Deep Sweep (L24-L27)
    try:
        with open('output/llama_deep_sweep.json', 'r') as f:
            d = json.load(f)
            base = d.get("Baseline", 0)
            
            for k, v in d.items():
                if k == "Baseline": continue
                parts = k.split('_') # L24_S-0.01
                layer = int(parts[0][1:])
                strength = float(parts[1][1:])
                
                cond = "Other"
                if strength == -0.01: cond = "Micro-Sharpen"
                elif strength == 0.01: cond = "Micro-Smooth"
                elif strength == 0.1: cond = "Mild-Smooth"
                
                data_points.append({"Layer": layer, "Condition": "Baseline", "Accuracy": base})
                data_points.append({"Layer": layer, "Condition": cond, "Accuracy": v})
    except: pass

    # 3. Mid Sweep (L13-L23) - CURRENTLY RUNNING
    try:
        with open('output/llama_mid_sweep.json', 'r') as f:
            d = json.load(f)
            base = d.get("Baseline", 0)
            
            for k, v in d.items():
                if k == "Baseline": continue
                parts = k.split('_')
                layer = int(parts[0][1:])
                strength = float(parts[1][1:])
                
                cond = "Other"
                if strength == -0.01: cond = "Micro-Sharpen"
                elif strength == 0.01: cond = "Micro-Smooth"
                elif strength == 0.1: cond = "Mild-Smooth"
                
                data_points.append({"Layer": layer, "Condition": "Baseline", "Accuracy": base})
                data_points.append({"Layer": layer, "Condition": cond, "Accuracy": v})
    except: pass

    if not data_points:
        print("No data found.")
        return

    df = pd.DataFrame(data_points)
    
    # Clean duplicates (Baseline repeated)
    # We want to plot curves for: Micro-Sharpen (-0.01) vs Micro-Smooth (+0.01) vs Mild-Smooth (+0.1)
    
    plt.figure(figsize=(12, 6))
    
    # Plot Baseline Line (Mean of baselines found)
    baselines = df[df["Condition"] == "Baseline"]["Accuracy"].unique()
    avg_base = baselines.mean()
    plt.axhline(y=avg_base, color='gray', linestyle='--', label=f'Baseline ({avg_base:.1%})')
    
    # Filter conditions
    conditions = ["Micro-Sharpen", "Micro-Smooth", "Mild-Smooth"]
    colors = ["red", "blue", "green"]
    markers = ["^", "o", "s"]
    
    for i, cond in enumerate(conditions):
        subset = df[df["Condition"] == cond].sort_values("Layer")
        if not subset.empty:
            plt.plot(subset["Layer"], subset["Accuracy"], marker=markers[i], label=cond, color=colors[i], linewidth=2)
            
    plt.title("Llama 3.2 3B: Topological Stability Profile (The J-Curve)")
    plt.xlabel("Layer Depth")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(12, 28))
    
    out_path = 'output/llama_j_curve.png'
    plt.savefig(out_path)
    print(f"J-Curve saved to {out_path}")

if __name__ == "__main__":
    main()
