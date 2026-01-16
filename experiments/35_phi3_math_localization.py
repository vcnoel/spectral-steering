import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re
from tqdm import tqdm

# --- Configuration ---
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
# We define layer blocks to scan (Phi-3 has 32 layers)
LAYER_BLOCKS = [
    (0, 8, "Early (Syntax)"),
    (8, 16, "Middle-Low (Semantics)"),
    (16, 24, "Middle-High (Reasoning)"),
    (24, 32, "Late (Output/Formatting)")
]

# We test two modes per block
CONFIGS = [
    {"alpha": 0.3, "name": "Smoothing (Blur)"},   # To see where it BREAKS math
    {"alpha": -0.2, "name": "Sharpening (Focus)"} # To see where it HELPS math
]

def apply_spectral_steering(model, start_layer, end_layer, alpha):
    """
    Applies spectral steering ONLY to a specific block of layers.
    """
    print(f"   -> Steering Layers {start_layer}-{end_layer} with Alpha {alpha}...")
    with torch.no_grad():
        for i in range(start_layer, end_layer):
            layer = model.model.layers[i]
            # Target the MLP Down Projection (usually the knowledge/reasoning store)
            weight = layer.mlp.down_proj.weight
            
            # SVD
            dtype = weight.dtype
            U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
            
            # Steering Formula
            # S_new = S * (1 - alpha)
            # If alpha is positive (0.3), S_new = S * 0.7 (Smoothing)
            # If alpha is negative (-0.2), S_new = S * 1.2 (Sharpening)
            
            S_new = S * (1.0 - alpha)
            
            # Reconstruct
            weight_new = (U @ torch.diag(S_new) @ Vh).to(dtype)
            layer.mlp.down_proj.weight.copy_(weight_new)

def reset_model_block(model, original_weights, start_layer, end_layer):
    """
    Restores the original weights for the specific block.
    """
    with torch.no_grad():
        for i in range(start_layer, end_layer):
            model.model.layers[i].mlp.down_proj.weight.copy_(original_weights[i].to(model.device))

def solve_math_problem(model, tokenizer, question):
    messages = [{"role": "user", "content": f"{question} Think step by step and give the final answer."}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        # Greedy for math stability
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False) 
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

def check_answer(response, ground_truth):
    # Standard GSM8K normalization (simple)
    target_str = ground_truth.split("####")[-1].strip()
    try:
        target_val = float(target_str.replace(',', ''))
    except:
        return False
        
    # Look for the number in the response
    # This regex handles integers and floats
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", response)
    
    for n in nums:
        try:
            val = float(n)
            # Exact match or close enough float
            if abs(val - target_val) < 1e-4:
                return True
        except:
            continue
            
    # Also check string overlap for safety (sometimes numbers are formatted weirdly)
    if target_str in response:
        return True
        
    return False

def main():
    print("Loading Phi-3...")
    # Use float16 for speed/memory on consumer GPU
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load small subset of GSM8K
    print("Loading GSM8K Subset...")
    dataset = load_dataset("gsm8k", "main", split="test")
    # Using 50 samples as requested
    dataset = dataset.select(range(50))

    # Save ALL original weights to RAM (for resets)
    # Target is MLP Down Proj on all 32 layers
    print("Backing up weights...")
    original_weights = {}
    for i in range(32):
        original_weights[i] = model.model.layers[i].mlp.down_proj.weight.clone().cpu()

    print("\n=== STARTING LAYER LOCALIZATION SWEEP ===")

    # 1. Baseline
    print("\n--- Baseline ---")
    correct = 0
    for item in tqdm(dataset, desc="Baseline"):
        ans = solve_math_problem(model, tokenizer, item['question'])
        if check_answer(ans, item['answer']):
            correct += 1
    baseline_acc = correct / len(dataset)
    print(f"Baseline Math Accuracy: {baseline_acc:.1%} ({correct}/{len(dataset)})")

    # 2. Block Sweeps
    results = []

    for start, end, label in LAYER_BLOCKS:
        print(f"\n--- Testing Block: {label} (Layers {start}-{end}) ---")
        
        for config in CONFIGS:
            alpha = config["alpha"]
            mode_name = config["name"]
            
            # Apply Steering
            apply_spectral_steering(model, start, end, alpha)
            
            # Eval
            correct = 0
            # Use tqdm for progress visibility
            for item in tqdm(dataset, desc=f"{label} {mode_name}"): 
                ans = solve_math_problem(model, tokenizer, item['question'])
                if check_answer(ans, item['answer']):
                    correct += 1
            
            acc = correct / len(dataset)
            delta = acc - baseline_acc
            
            print(f"   [{mode_name}] Alpha {alpha} -> Accuracy: {acc:.1%} (Delta: {delta:+.1%})")
            results.append({
                "block": label, 
                "layers": f"{start}-{end}",
                "mode": mode_name, 
                "alpha": alpha,
                "acc": acc,
                "delta": delta
            })
            
            # Reset specific block
            reset_model_block(model, original_weights, start, end)

    print("\n=== FINAL LOCALIZATION MAP ===")
    print(f"{'Block':<25} | {'Mode':<20} | {'Acc':<8} | {'Delta':<8}")
    print("-" * 70)
    for res in results:
        print(f"{res['block']:<25} | {res['mode']:<20} | {res['acc']:.1%} | {res['delta']:+.1%}")

if __name__ == "__main__":
    main()
