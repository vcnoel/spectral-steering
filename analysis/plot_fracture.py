import json
import matplotlib.pyplot as plt
import os

def main():
    # Load data
    with open('output/fracture_log.json', 'r') as f:
        data = json.load(f)

    plt.figure(figsize=(10, 6))
    
    # Styles for differentiation
    styles = {
        "healthy_baseline": {"color": "blue", "label": "Healthy (Baseline)", "style": "-"},
        "passive_voice_scar": {"color": "red", "label": "Fracture (Passive Voice)", "style": "--"},
        "logic_trap": {"color": "orange", "label": "Fracture (Logic)", "style": "-."}
    }

    for entry in data:
        etype = entry['type']
        if etype not in styles: continue
        
        y = entry['fiedler_profile']
        x = range(len(y))
        
        plt.plot(x, y, 
                label=styles[etype]["label"], 
                color=styles[etype]["color"], 
                linestyle=styles[etype]["style"],
                linewidth=2.5)

    # Annotation for the "Snap"
    plt.annotate('Topological Fracture', xy=(12, 0.05), xytext=(15, 0.3),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12)

    plt.title("Spectral Signature of Reasoning Collapse (Phi-3.5-mini)", fontsize=14)
    plt.xlabel("Layer Depth", fontsize=12)
    plt.ylabel("Algebraic Connectivity (Fiedler Value)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs('analysis', exist_ok=True)
    plt.savefig('analysis/fracture_plot.png', dpi=300)
    print("Plot saved to analysis/fracture_plot.png")

if __name__ == "__main__":
    main()
