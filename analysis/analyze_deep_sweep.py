import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def main():
    path = 'output/llama_deep_sweep.json'
    if not os.path.exists(path):
        print("No results yet.")
        return

    with open(path, 'r') as f:
        data = json.load(f)

    # Convert to DataFrame
    # Format: {"Baseline": acc, "L24_S-0.01": acc, ...}
    
    rows = []
    baseline = data.get("Baseline", 0)
    
    for k, v in data.items():
        if k == "Baseline": continue
        
        # Parse Key: L{layer}_S{strength}
        try:
            parts = k.split('_')
            layer = int(parts[0][1:])
            strength = float(parts[1][1:])
            
            rows.append({
                "Layer": layer,
                "Strength": strength,
                "Accuracy": v,
                "Delta": v - baseline
            })
        except:
            pass

    if not rows:
        print(f"Only Baseline found: {baseline:.2%}")
        return

    df = pd.DataFrame(rows)
    print(f"Baseline: {baseline:.2%}")
    print(df.sort_values("Accuracy", ascending=False))

    # heat map pivot
    pivot = df.pivot(index="Layer", columns="Strength", values="Accuracy")
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".1%", cmap="RdYlGn", center=baseline)
    plt.title(f"Llama 3.2 Deep Sweep (Baseline: {baseline:.1%})")
    plt.tight_layout()
    plt.savefig('output/llama_deep_sweep_heatmap.png')
    print("Heatmap saved to output/llama_deep_sweep_heatmap.png")

if __name__ == "__main__":
    main()
