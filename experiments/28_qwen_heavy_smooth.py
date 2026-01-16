import torch
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import numpy as np

# CONFIG
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TARGET_LAYER = 27 # Final Layer
STRENGTH = 0.5    # Requested Heavy Smoothing

def get_smoothness_grad(hidden):
    """Minimizes Dirichlet Energy (Maximizes Smoothness) on Self-Attention Graph"""
    hidden = hidden.detach().requires_grad_(True)
    with torch.enable_grad():
        X = hidden.squeeze(0).float()
        # Dynamic Graph from Self-Attention
        A = torch.softmax((X @ X.T) / (X.shape[-1]**0.5), dim=-1)
        D = torch.diag(A.sum(1))
        L = D - A
        energy = torch.trace(X.T @ L @ X)
        grad = torch.autograd.grad(energy, hidden)[0]
    return -grad # This is the SMOOTHING direction

def main():
    print(f"Loading {MODEL_NAME} for Heavy Smooth Check (L{TARGET_LAYER}, S={STRENGTH})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    
    with open('data/logic_dataset.json', 'r') as f:
        data = json.load(f) # Full N=300

    results = {}

    # 1. BASELINE
    print("\n--- Running Baseline ---")
    success_count = 0
    for item in tqdm(data):
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
            if target in gen: success_count += 1
            elif "negation_trap" in item['type'] and "not" in gen: success_count += 1
        except Exception as e: print(e)
    
    baseline_acc = success_count / len(data)
    results["Baseline"] = baseline_acc
    print(f"Result Baseline: {baseline_acc:.2%}")

    # 2. HEAVY SMOOTH RUN
    config_name = f"L{TARGET_LAYER}_S{STRENGTH}"
    print(f"\n--- Running {config_name} ---")
            
    def hook(module, input_args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        
        # Calculate Spec Vector (Smoothing Direction)
        spec_vec = get_smoothness_grad(hidden)
        spec_norm = torch.norm(spec_vec)
        
        perturbation = torch.zeros_like(hidden)
        if spec_norm > 1e-6:
            perturbation = spec_vec / spec_norm * torch.norm(hidden) * 0.1 * STRENGTH
            
        return (hidden + perturbation,) + output[1:] if isinstance(output, tuple) else (hidden + perturbation)

    handle = model.model.layers[TARGET_LAYER].register_forward_hook(hook)
    
    # Eval
    success_count = 0
    for item in tqdm(data):
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
            if target in gen: success_count += 1
            elif "negation_trap" in item['type'] and "not" in gen: success_count += 1
        except Exception as e: print(e)
    
    acc = success_count / len(data)
    results[config_name] = acc
    print(f"Result {config_name}: {acc:.2%}")
    
    handle.remove()
    
    # Save Results
    with open("output/qwen_heavy_smooth.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\nHeavy Smooth Check Complete.")

if __name__ == "__main__":
    main()
