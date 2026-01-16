import json
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_FILE = "output/full_spectral_profile.json"
PLOT_FILE = "analysis/phase_space_plot.png"

def main():
    try:
        with open(OUTPUT_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {OUTPUT_FILE} not found. Run experiments/11_full_spectral_profile.py first.")
        return

    plt.figure(figsize=(10, 8))
    
    colors = {"Healthy": "blue", "Nested": "red"}
    styles = {"Healthy": "-o", "Nested": "-x"}

    for item in data:
        label_key = item['type'].split(" ")[0] # "Nested" or "Healthy"
        label = item['type']
        
        metrics = item['metrics']
        fiedler = metrics['fiedler_profile']
        entropy = metrics['entropy_profile']
        
        c = colors.get(label_key, "gray")
        s = styles.get(label_key, "-")
        
        # Plot Trajectory
        plt.plot(fiedler, entropy, s, label=label, color=c, linewidth=2, markersize=6, alpha=0.8)
        
        # Annotate Key Layers
        # Start (Layer 0)
        plt.text(fiedler[0], entropy[0], "L0", fontsize=10, color=c, fontweight='bold')
        
        # Fracture Point (Layer 2)
        plt.text(fiedler[2], entropy[2], "L2", fontsize=10, color=c, fontweight='bold')
        
        # End (Layer 31)
        plt.text(fiedler[-1], entropy[-1], "L31", fontsize=10, color=c, fontweight='bold')

        # Arrows to show direction
        for i in range(0, len(fiedler)-1, 4):
            plt.arrow(fiedler[i], entropy[i], fiedler[i+1]-fiedler[i], entropy[i+1]-entropy[i], 
                      shape='full', lw=0, length_includes_head=True, head_width=0.01, color=c)

    plt.title("Spectral Phase Space: Topology vs Complexity")
    plt.xlabel("Algebraic Connectivity (Fiedler Value) $\\lambda_2$")
    plt.ylabel("Spectral Entropy (Complexity)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # Add Zone Annotations
    plt.axvline(x=0.2, color='gray', linestyle=':', alpha=0.5)
    plt.text(0.1, np.mean(plt.ylim()), "Fracture Zone\n(Disconnected)", ha='center', color='gray')
    
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300)
    print(f"Phase Space plot saved to {PLOT_FILE}")

if __name__ == "__main__":
    main()
