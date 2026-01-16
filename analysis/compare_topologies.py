import matplotlib.pyplot as plt
import numpy as np

# Data from Sensitivity Scans ("The Shake Test")
# Normalized Layer Index (0.0 to 1.0) for fair comparison
# Accuracy normalized to Baseline (Acc / Baseline_Acc) to show relative drop

# PHI-3 MINI (3.8B) - The "Deep Synthesizer" & "Void"
# Baseline ~78%. L2 drops to ~10%. L24 drops to ~50% (Cliff).
phi3_layers = np.linspace(0, 1, 32)
phi3_sensitivity = np.ones(32) * 0.9  # General noise
phi3_sensitivity[2:5] = 0.15 # The Void (L2-L4)
phi3_sensitivity[20:25] = 0.6 # The Cliff (L24)

# QWEN 2.5 (1.5B) - The "Early Crunch"
# Baseline ~52%. L7 drops to 22% (0.42 relative).
qwen_layers = np.linspace(0, 1, 28)
qwen_metrics = [
    0.86, 0.76, 0.40, 0.46, 0.36, 0.36, 0.40, 
    0.22, 0.22, 0.26, 0.38, 0.42, 0.48, 0.40,
    0.44, 0.64, 0.64, 0.62, 0.74, 0.74, 0.72,
    0.62, 0.72, 0.62, 0.60, 0.58, 0.58, 0.64
]
# Normalize by max/baseline to get "Relative Retention"
qwen_rel = np.array(qwen_metrics) / 0.86

# LLAMA 3.2 (3B) - The "Void" (Similar to Phi-3)
# Baseline ~? (Let's assume ~60% for now). L2 drops to 10% (0.16 relative).
llama_layers = np.linspace(0, 1, 28) # Llama 3.2 3B has 28 layers
# Data from scan (Step 1492):
# L0:34, L1:20, L2:10, L3:22, L4:30, L5:30, L6:26, L7:34...
llama_metrics = [
    0.34, 0.20, 0.10, 0.22, 0.30, 0.30, 0.26, 0.34,
    0.24, 0.36, 0.42, 0.42, 0.36, 0.42, 0.38, 0.42,
    0.40, 0.36, 0.38, 0.42, 0.42, 0.64, 0.40, 0.40,
    0.36, 0.40, 0.42, 0.40
]
# Normalize (assuming L0/L10 peak is reference)
llama_baseline = max(llama_metrics)
llama_rel = np.array(llama_metrics) / llama_baseline

def plot_comparison():
    plt.figure(figsize=(12, 6))
    
    # Plot Phi-3 (Smooth curve)
    # plt.plot(phi3_layers, phi3_sensitivity, label='Phi-3 Mini (3.8B)', color='gray', linestyle='--', alpha=0.5)
    
    # Plot Qwen
    plt.plot(qwen_layers, qwen_rel, label='Qwen 2.5 (1.5B)', color='crimson', linewidth=2, marker='o', markersize=4)
    
    # Plot Llama
    plt.plot(llama_layers, llama_rel, label='Llama 3.2 (3B)', color='royalblue', linewidth=2, marker='s', markersize=4)

    plt.title("The Universal Topological Void: Sensitivity across SLMs", fontsize=14)
    plt.xlabel("Normalized Layer Depth (0=Input, 1=Output)", fontsize=12)
    plt.ylabel("Functional Retention (Normalized Accuracy)", fontsize=12)
    
    plt.axvline(x=7/28, color='crimson', linestyle=':', alpha=0.5, label='Qwen Brittle Hub (L7)')
    plt.axvline(x=2/28, color='royalblue', linestyle=':', alpha=0.5, label='Llama Brittle Hub (L2)')
    
    plt.text(7/28 + 0.02, 0.3, "Qwen Hub", color='crimson')
    plt.text(2/28 + 0.02, 0.15, "Llama Voids", color='royalblue')

    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    
    plt.savefig("output/slm_landscape_comparison.png")
    print("Saved output/slm_landscape_comparison.png")

if __name__ == "__main__":
    plot_comparison()
