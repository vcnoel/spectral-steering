
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re
from tqdm import tqdm
import json
import os
import gc

# --- Configuration ---
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
TARGET_LAYER = 15
ALPHAS = [-0.05, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.8]  # Sharpening range
N_SAMPLES = 100
OUTPUT_FILE = "output/l15_alpha_sweep_results.json"

def apply_spectral_steering_single_layer(model, layer_idx, alpha, original_weight):
    """
    Applies spectral steering to a SINGLE layer (SVD Scaling).
    S_new = S * (1 - alpha).
    Negative alpha -> S * (1 + |alpha|) -> Sharpening.
    """
    with torch.no_grad():
        layer = model.model.layers[layer_idx]
        weight = layer.mlp.down_proj.weight
        
        # SVD
        dtype = weight.dtype
        # Use float for SVD
        U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
        
        # Steering
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
    total = 0
    
    # We use a simple progress bar
    for item in tqdm(dataset, desc="Eval", leave=False):
        # Phi-3 Prompting
        messages = [{"role": "user", "content": f"{item['question']} Think step by step."}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256, 
                do_sample=False, 
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Extract target
        target_str = item['answer'].split("####")[-1].strip()
        try:
            target_val = float(target_str.replace(',', ''))
        except:
            total += 1
            continue
        
        # Extract prediction
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", response)
        # Check last number first? Or any matching?
        # Standard GSM8K usually takes the lat one.
        # But robust extraction checks if *any* number matches (sometimes beneficial) 
        # or stricly the last. The previous script logic was:
        # for n in nums: if match break. This is lenient. I will stick to it for consistency.
        
        found = False
        for n in nums:
            try:
                if abs(float(n) - target_val) < 1e-4:
                    correct += 1
                    found = True
                    break
            except:
                continue
        total += 1
    
    return correct / total if total > 0 else 0

def compute_ppl(model, tokenizer, text="The quick brown fox jumps over the lazy dog. " * 10):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        ppl = torch.exp(outputs.loss).item()
    return ppl

def main():
    print(f"=== L{TARGET_LAYER} ALPHA SWEEP (Phi-3) ===")
    
    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load Data
    print(f"Loading GSM8K (N={N_SAMPLES})...")
    dataset = load_dataset("gsm8k", "main", split="test").select(range(N_SAMPLES))
    
    # Backup Weight
    print(f"Backing up weight for Layer {TARGET_LAYER}...")
    original_weight = model.model.layers[TARGET_LAYER].mlp.down_proj.weight.clone().cpu()
    
    results = {}
    
    # Baseline
    print("\n--- Baseline ---")
    baseline_acc = evaluate_math(model, tokenizer, dataset)
    baseline_ppl = compute_ppl(model, tokenizer)
    print(f"Baseline: Acc={baseline_acc:.1%}, PPL={baseline_ppl:.2f}")
    
    results["baseline"] = {"acc": baseline_acc, "ppl": baseline_ppl}
    
    # Sweep
    for alpha in ALPHAS:
        print(f"\n--- Testing Alpha {alpha} ---")
        
        # Apply Steering
        apply_spectral_steering_single_layer(model, TARGET_LAYER, alpha, original_weight)
        
        # Eval
        acc = evaluate_math(model, tokenizer, dataset)
        ppl = compute_ppl(model, tokenizer)
        
        delta = acc - baseline_acc
        print(f"Alpha {alpha}: Acc={acc:.1%} (Δ {delta:+.1%}), PPL={ppl:.2f}")
        
        results[str(alpha)] = {
            "acc": acc, 
            "delta": delta,
            "ppl": ppl
        }
        
        # Reset
        reset_layer(model, TARGET_LAYER, original_weight)
        
    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
