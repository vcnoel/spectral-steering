import matplotlib.pyplot as plt
import numpy as np
import os

OUTPUT_DIR = "figure_papers"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def plot_model_zoo():
    print("Generating Plot 3: The Model Zoo (Topological Comparison)...")
    
    # Metrics to Compare:
    # 1. Logic Fracture Repair (Delta Accuracy with Steering)
    # 2. PPL Sensitivity (Log PPL increase at alpha=0.5)
    # 3. Steerability Score (Subjective 0-10 based on Phase Space)
    
    models = ['Phi-3 Mini', 'Qwen 2.5', 'Llama 3.2']
    
    # 1. Logic Repair (Delta)
    # Phi-3: 51% -> 78% (+27%)
    # Qwen: 51% -> 50% (0%) - Robust/Safe
    # Llama: 36% -> 3% (-33%) - Inverted/Broken
    logic_delta = [27.0, 0.0, -33.0]
    
    # 2. PPL Sensitivity (at alpha=0.5)
    # Phi-3: PPL 5 -> 67 (+60)
    # Qwen: PPL 14 -> 14 (0)
    # Llama: PPL 3 -> 3 (0)
    ppl_impact = [60.0, 0.5, 0.1]
    
    # 3. Peak Math Steerability (Delta) - if applicable
    # Phi-3: +1.3% (Spectra-Phi Max)
    # Qwen: Unknown/Stable? (Assume 0 for now)
    # Llama: Likely 0 or Negative (Diamond)
    math_delta = [1.3, 0.0, 0.0] 

    x = np.arange(len(models))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Bar 1: Logic Repair (Left Axis)
    rects1 = ax1.bar(x - width/2, logic_delta, width, label='Logic Repair (Delta %)', color='skyblue', alpha=0.8)
    
    # Bar 2: PPL Sensitivity (Right Axis?) No, let's keep it simple.
    # Let's just plot Logic Repair as the main "Steerability" proxy.
    
    # Actually, let's plot "Steerability Profile"
    # X-Axis: Model
    # Y-Axis: Score
    
    # Let's do a "Radar Chart" instead? Or just a simple bar chart of "Responsiveness".
    # Let's stick to the Logic Delta as it's the most striking visual.
    
    ax1.set_ylabel('Steering Impact (Accuracy Delta %)', fontsize=12)
    ax1.set_title('The Topological Zoo: Steerability by Model', fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=12)
    ax1.axhline(0, color='black', linewidth=1)
    
    # Add labels
    ax1.bar_label(rects1, padding=3, fmt='%+.1f%%')
    
    # Annotations
    ax1.annotate("The Glass Cannon\n(Highly Steerable)", xy=(0, 27), xytext=(0, 35), 
                 ha='center', arrowprops=dict(facecolor='black', shrink=0.05))
                 
    ax1.annotate("The Tank\n(Robust)", xy=(1, 0), xytext=(1, 10), 
                 ha='center', arrowprops=dict(facecolor='black', shrink=0.05))
                 
    ax1.annotate("The Diamond\n(Unsteerable)", xy=(2, -33), xytext=(2, -20), 
                 ha='center', arrowprops=dict(facecolor='black', shrink=0.05))

    ax1.set_ylim(-40, 45)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right')
    
    plt.savefig(os.path.join(OUTPUT_DIR, "model_zoo_comparison.png"), dpi=300)
    print("Model Zoo Plot saved.")

if __name__ == "__main__":
    plot_model_zoo()
