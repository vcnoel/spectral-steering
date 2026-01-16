# Spectral Steering

**Spectral Steering** is a technique to precisely adjust the behavior of Large Language Models (LLMs) by modifying the singular value spectrum of specific weight matrices (typically `mlp.down_proj`).

Unlike fine-tuning, it requires **zero gradient updates**. It operates by identifying "noisy" or "competent" spectral bands and sharpening (negative alpha) or smoothing (positive alpha) them.

## Supported Models & Key Results

We have benchmarked Spectral Steering across multiple model families. Key findings include:

### 1. Llama 3.2 3B (The "Pareto" Winner)
We identified a **Global Optimal Configuration** that contradicts the typical trade-off between safety and capability.

*   **Config:** `L24_A-0.3` (Layer 24, Alpha -0.3, Sharpening).
*   **Safety (Sycophancy):** **+2.4%** gain (77.4% -> 79.8%).
*   **Reasoning (GSM8K):** **+1.4%** gain (64.3% -> 65.7%).
*   **Conclusion:** Sharpening late-layer gradients improves both instruction following and logical precision.

### 2. Phi-3 Mini (The "Hybrid" Strategy)
Phi-3 required a more complex, multi-stage intervention to balance its highly condensed weights.

*   **Config:** **Unified Hybrid**
    *   **Layer 15:** Alpha -0.2 (Kick/Sharpening)
    *   **Layers 16-24:** Alpha +0.3 (Smoothing/Stabilization)
*   **Results:**
    *   **Sycophancy:** Reduced by **~27%** relative (0.30 -> 0.22).
    *   **Math:** Slight improvement (+1%).

### 3. Llama 3.1 8B
Validated the importance of the "Mid-Layers" (Layer 20 specifically) for safety interventions.
*   **Config:** Layer 20, Alpha 0.5 (Smoothing).
*   **Result:** Strong reduction in sycophancy with minimal impact on perplexity.

### 4. Qwen 2.5 0.5B (The "Robust" Model)
Qwen models exhibited significant resistance to spectral steering, adhering to a "Universal Law of Robustness" where weight spectra are harder to shift beneficially without retraining.
*   **Result:** Neutral/Invariant.

## Repository Structure

*   `experiments/`: Python scripts for all sweeps and validations.
    *   **Llama 3.2:** `66_llama_3_2_targeted_sweep.py`, `68_llama_3_2_extremes_confirmation.py`
    *   **Phi-3:** `43_phi3_unified_final.py`
    *   **Qwen:** `27_qwen_universal_law.py`
*   `notebooks/`: Analysis and Visualization.
    *   `Analyze_Llama_3_2_Results.ipynb`: **Start Here** for Llama 3.2 analysis.
*   `output/`: JSON/CSV logs of all experimental runs.

## Usage

1.  **Installation:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run an Experiment:**
    ```bash
    python experiments/68_llama_3_2_extremes_confirmation.py
    ```
3.  **Analyze:**
    Open `notebooks/Analyze_Llama_3_2_Results.ipynb` in Jupyter/VSCode.

## License

MIT
