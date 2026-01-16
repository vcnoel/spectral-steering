import torch
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import numpy as np

# CONFIG
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
LAYERS_TO_TEST = [24, 25, 26, 27]
STRENGTHS = [-0.01, 0.01, 0.1] 

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
    return -grad # This is the SMOOTHING direction (Gradient Descent on Energy)

def main():
    print(f"Loading {MODEL_NAME} for Deep Sweep...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

    with open('data/logic_dataset.json', 'r') as f:
        data = json.load(f) # Full N=300

    results = {}

    # 1. BASELINE (Re-run to be safe and local)
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

    # 2. SWEEP LOOP
    for layer in LAYERS_TO_TEST:
        for strength in STRENGTHS:
            config_name = f"L{layer}_S{strength}"
            print(f"\n--- Running {config_name} ---")
            
            def hook(module, input_args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                
                # Calculate Spec Vector (Smoothing Direction)
                spec_vec = get_smoothness_grad(hidden)
                spec_norm = torch.norm(spec_vec)
                
                perturbation = torch.zeros_like(hidden)
                if spec_norm > 1e-6:
                    # Perturbation = vec * strength
                    # If strength is NEGATIVE (-0.01), we move AGAINST Smoothing (Sharpening).
                    # If strength is POSITIVE (+0.01), we move TOWARDS Smoothing.
                    # Normalization: Scaled relative to hidden state norm (0.1 factor)
                    perturbation = spec_vec / spec_norm * torch.norm(hidden) * 0.1 * strength
                    
                return (hidden + perturbation,) + output[1:] if isinstance(output, tuple) else (hidden + perturbation)

            handle = model.model.layers[layer].register_forward_hook(hook)
            
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
            
            # Incremental Save
            if not os.path.exists('output'): os.makedirs('output')
            with open("output/llama_deep_sweep.json", 'w') as f:
                json.dump(results, f, indent=2)

    print("\nDeep Sweep Complete.")

if __name__ == "__main__":
    main()
