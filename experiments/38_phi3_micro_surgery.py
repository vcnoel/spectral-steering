"""
Phase 25: Micro-Surgery (Layer-by-Layer within Glass Jaw)

Goal: Find the exact layer $L$ where Sharpening switches from "Useless" to "Helpful."

Hypothesis:
- Layers 8-11: "Semantic Setup" (Sensitivity)
- Layers 12-16: "Reasoning Initiation" (Calculation)

Method: Layer-by-layer sweep on L8-L16 with Sharpening (α=-0.2), N=100 GSM8K.
Metrics:
- Math Accuracy (primary signal)
- PPL (safety signal)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re
from tqdm import tqdm
import json

# --- Configuration ---
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
TARGET_LAYERS = list(range(8, 17))  # Layers 8-16 inclusive
ALPHA = -0.2  # Sharpening
N_SAMPLES = 100  # High N for precise signal

def apply_spectral_steering_single_layer(model, layer_idx, alpha, original_weight):
    """
    Applies spectral steering to a SINGLE layer.
    """
    with torch.no_grad():
        layer = model.model.layers[layer_idx]
        weight = layer.mlp.down_proj.weight
        
        # SVD
        dtype = weight.dtype
        U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
        
        # Steering: S_new = S * (1 - alpha)
        S_new = S * (1.0 - alpha)
        
        # Reconstruct
        weight_new = (U @ torch.diag(S_new) @ Vh).to(dtype)
        layer.mlp.down_proj.weight.copy_(weight_new)

def reset_layer(model, layer_idx, original_weight):
    """
    Restores the original weight for a single layer.
    """
    with torch.no_grad():
        model.model.layers[layer_idx].mlp.down_proj.weight.copy_(original_weight.to(model.device))

def evaluate_math(model, tokenizer, dataset):
    """
    Evaluates Math accuracy on GSM8K.
    """
    correct = 0
    
    for item in tqdm(dataset, desc="Math", leave=False):
        messages = [{"role": "user", "content": f"{item['question']} Think step by step."}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Extract answer from GSM8K format
        target_str = item['answer'].split("####")[-1].strip()
        try:
            target_val = float(target_str.replace(',', ''))
        except:
            continue
        
        # Look for the target in response
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", response)
        for n in nums:
            try:
                if abs(float(n) - target_val) < 1e-4:
                    correct += 1
                    break
            except:
                continue
    
    return correct / len(dataset)

def compute_ppl(model, tokenizer, text="The quick brown fox jumps over the lazy dog. " * 10):
    """
    Computes perplexity on a simple text.
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        ppl = torch.exp(outputs.loss).item()
    return ppl

def main():
    print("Loading Phi-3...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load GSM8K
    print(f"Loading GSM8K (N={N_SAMPLES})...")
    dataset = load_dataset("gsm8k", "main", split="test").select(range(N_SAMPLES))
    
    # Backup ALL target layer weights
    print("Backing up weights for layers 8-16...")
    original_weights = {}
    for i in TARGET_LAYERS:
        original_weights[i] = model.model.layers[i].mlp.down_proj.weight.clone().cpu()
    
    # 1. Baseline
    print("\n=== BASELINE ===")
    baseline_math = evaluate_math(model, tokenizer, dataset)
    baseline_ppl = compute_ppl(model, tokenizer)
    print(f"Baseline Math: {baseline_math:.1%}")
    print(f"Baseline PPL: {baseline_ppl:.2f}")
    
    # 2. Layer-by-Layer Sweep
    results = []
    
    print(f"\n=== MICRO-SURGERY SWEEP (Sharpening α={ALPHA}) ===")
    for layer_idx in TARGET_LAYERS:
        print(f"\n--- Layer {layer_idx} ---")
        
        # Apply sharpening to THIS layer only
        apply_spectral_steering_single_layer(model, layer_idx, ALPHA, original_weights[layer_idx])
        
        # Evaluate
        math_acc = evaluate_math(model, tokenizer, dataset)
        ppl = compute_ppl(model, tokenizer)
        delta_math = math_acc - baseline_math
        
        print(f"   Math: {math_acc:.1%} (Δ {delta_math:+.1%})")
        print(f"   PPL: {ppl:.2f}")
        
        results.append({
            "layer": layer_idx,
            "math_acc": math_acc,
            "delta_math": delta_math,
            "ppl": ppl
        })
        
        # Reset this layer before moving to next
        reset_layer(model, layer_idx, original_weights[layer_idx])
    
    # 3. Final Report
    print("\n" + "="*60)
    print("MICRO-SURGERY RESULTS: Layer-by-Layer Sharpening")
    print("="*60)
    print(f"{'Layer':<10} | {'Math Acc':<10} | {'Delta':<10} | {'PPL':<10}")
    print("-" * 50)
    print(f"{'Baseline':<10} | {baseline_math:.1%} | {'—':^10} | {baseline_ppl:.2f}")
    print("-" * 50)
    
    for res in results:
        delta_str = f"{res['delta_math']:+.1%}"
        print(f"L{res['layer']:<9} | {res['math_acc']:.1%} | {delta_str:^10} | {res['ppl']:.2f}")
    
    # 4. Find the crossover point
    print("\n=== ANALYSIS ===")
    best_layer = max(results, key=lambda x: x['delta_math'])
    worst_layer = min(results, key=lambda x: x['delta_math'])
    
    print(f"Best Layer for Sharpening: L{best_layer['layer']} ({best_layer['delta_math']:+.1%})")
    print(f"Worst Layer for Sharpening: L{worst_layer['layer']} ({worst_layer['delta_math']:+.1%})")
    
    # Save results
    output_path = "output/micro_surgery_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "baseline_math": baseline_math,
            "baseline_ppl": baseline_ppl,
            "alpha": ALPHA,
            "n_samples": N_SAMPLES,
            "results": results
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
