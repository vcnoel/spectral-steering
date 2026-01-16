import json
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_FILE = "output/full_spectral_profile.json"
PLOT_FILE = "analysis/full_spectral_metrics.png"

def main():
    with open(OUTPUT_FILE, 'r') as f:
        data = json.load(f)
        
    # Metrics to plot
    metrics = [
        ("fiedler_profile", "Algebraic Connectivity (Fiedler Value)", "Connectivity"),
        ("smoothness_profile", "Smoothness Index (Norm. Dirichlet Energy)", "Smoothness"),
        ("entropy_profile", "Spectral Entropy", "Complexity"),
        ("hfer_profile", "High Frequency Energy Ratio (HFER)", "Noise")
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = {"Healthy": "blue", "Passive": "red", "Nested": "orange"}
    styles = {"Healthy": "-", "Passive": "--", "Nested": "-."}

    for i, (key, title, ylabel) in enumerate(metrics):
        ax = axes[i]
        
        for item in data:
            label = item['type'].split(" ")[0] # "Passive", "Nested", "Healthy"
            y_values = item['metrics'][key]
            x_values = range(len(y_values))
            
            c = colors.get(label, "gray")
            s = styles.get(label, "-")
            
            ax.plot(x_values, y_values, label=item['type'], color=c, linestyle=s, linewidth=2)
            
        ax.set_title(title)
        ax.set_xlabel("Layer Depth")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
            
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300)
    print(f"Plot saved to {PLOT_FILE}")

if __name__ == "__main__":
    main()
