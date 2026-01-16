# Spectral Steering for Llama 3.2

This repository contains experiments and analysis for **Spectral Steering**, a technique to adjust model behavior (e.g., Sycophancy, Reasoning) by modifying the singular value spectrum of specific layers.

## Key Findings (Llama 3.2 3B)

We identified a **Global Optimal Configuration** that improves both Safety and Reasoning without regression:

**Configuration:** `L24_A-0.3` (Layer 24, Alpha -0.3, Sharpening)
*   **Sycophancy (Safety):** +2.4%
*   **Math (GSM8K):** +1.4%
*   **Perplexity:** Neutral

## Repository Structure

*   `experiments/`: Scripts for conducting sweeps and validation runs.
    *   `66_llama_3_2_targeted_sweep.py`: Targeted Alpha/Layer sweep.
    *   `67_llama_3_2_final_confirmation.py`: Full-scale validation script.
    *   `68_llama_3_2_extremes_confirmation.py`: Verification of extreme candidates.
*   `notebooks/`: Analysis notebooks.
    *   `Analyze_Llama_3_2_Results.ipynb`: Visualizations of the results (Pareto Frontier, Champions Bar Chart).
*   `output/`: CSV logs of experiment results.

## Usage

1.  Install dependencies (transformers, torch, datasets, accelerate).
2.  Run an experiment script:
    ```bash
    python experiments/68_llama_3_2_extremes_confirmation.py
    ```
3.  Analyze results using the provided notebook.

## License

MIT
