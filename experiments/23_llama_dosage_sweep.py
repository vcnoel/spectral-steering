import torch
import json
import os
import argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# CONFIG
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_FILE = "output/llama_dosage_sweep.json"
DATA_FILE = "data/logic_dataset.json"
TARGET_LAYER = 12
DOSAGES = [-0.01, -0.05, -0.1, -0.2, -0.5] 

def get_smoothness_grad(hidden):
    """
    Minimizes Dirichlet Energy on the Intrinsic Self-Attention Graph.
    (This matches experiments/19 logic which yielded 22% survival)
    """
    hidden = hidden.detach().requires_grad_(True)
    
    with torch.enable_grad():
        X = hidden.squeeze(0).float()
        # Create Dynamic Adjacency from Self-Attention
        # This respects the model's internal topology instead of forcing uniformity
        A = torch.softmax((X @ X.T) / (X.shape[-1]**0.5), dim=-1)
        D = torch.diag(A.sum(1))
        L = D - A
        energy = torch.trace(X.T @ L @ X)
        grad = torch.autograd.grad(energy, hidden)[0]
        
    return -grad # Returns Smoothing Vector

def main():
    if not os.path.exists('output'): os.makedirs('output')
    
    print(f"Loading {MODEL_NAME} for Dosage Sweep @ L{TARGET_LAYER}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

    with open(DATA_FILE, 'r') as f:
        full_dataset = json.load(f)
        
    # Full N=300 for reliability
    eval_set = full_dataset 
    print(f"Evaluating on {len(eval_set)} samples.")

    results = {}
    
    # 1. Establish Baseline
    print("Running Baseline...")
    success_count = 0
    for item in tqdm(eval_set):
        prompt = item['prompt']
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        try:
            input_len = inputs.input_ids.shape[1]
            out = model.generate(**inputs, max_new_tokens=15, pad_token_id=tokenizer.eos_token_id)
            gen_ids = out[0][input_len:]
            gen = tokenizer.decode(gen_ids, skip_special_tokens=True).lower()
            
            target = item.get('target_completion', item.get('target_word', 'wug')).lower()
            if target in gen:
                success_count += 1
            elif "negation_trap" in item['type'] and "not" in gen:
                success_count += 1
        except Exception:
            pass
            
    base_acc = success_count / len(eval_set)
    results["Baseline"] = base_acc
    print(f"Baseline Accuracy: {base_acc:.2%}")

    # 2. Run Dosage Sweep
    for strength in DOSAGES:
        condition_name = f"L{TARGET_LAYER}_Strength_{strength}"
        print(f"\n--- Running {condition_name} ---")
        
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            
            # Get Smoothing Gradient (Dynamic Graph)
            spec_vec = get_smoothness_grad(hidden)
            spec_norm = torch.norm(spec_vec)
            
            perturbation = torch.zeros_like(hidden)
            
            if spec_norm > 1e-6:
                # Match Normalization from Phase 8 (Exp 19)
                # target_norm = 0.1 * hidden_norm * strength
                # Here strength is passed directly (e.g. -0.01)
                # So if strength is negative, perturbation opposes spec_vec (Sharpening)
                perturbation = spec_vec / spec_norm * torch.norm(hidden) * 0.1 * strength
            
            return (hidden + perturbation,) + output[1:] if isinstance(output, tuple) else (hidden + perturbation)
            
        handle = model.model.layers[TARGET_LAYER].register_forward_hook(hook)
        
        success_count = 0
        for item in tqdm(eval_set):
            prompt = item['prompt']
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            try:
                input_len = inputs.input_ids.shape[1]
                out = model.generate(**inputs, max_new_tokens=15, pad_token_id=tokenizer.eos_token_id)
                gen_ids = out[0][input_len:]
                gen = tokenizer.decode(gen_ids, skip_special_tokens=True).lower()
                
                target = item.get('target_completion', item.get('target_word', 'wug')).lower()
                if target in gen:
                    success_count += 1
                elif "negation_trap" in item['type'] and "not" in gen:
                    success_count += 1
            except Exception:
                pass
        
        handle.remove()
        
        acc = success_count / len(eval_set)
        results[condition_name] = acc
        print(f"Result {condition_name}: {acc:.2%}")
        
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2)

    print("\nDosage Sweep Complete.")

if __name__ == "__main__":
    main()
