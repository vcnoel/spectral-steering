import torch
import json
import os
import argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from spectral_trust import GSPConfig, GraphConstructor
from tqdm import tqdm

# CONFIG
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_FILE = "output/llama_sharpening_sweep.json"
DATA_FILE = "data/logic_dataset.json"
LAYERS = [2, 6, 8, 10, 12, 14, 16, 20] # Exploring the "Thinking" Middle Layers
STRENGTH = -1.0 # Sharpening (Gradient Ascent)

def get_spectral_gradient(hidden_states, current_attn_matrix, graph_engine):
    hidden_states = hidden_states.detach().requires_grad_(True)
    with torch.enable_grad():
        device = hidden_states.device
        dtype = hidden_states.dtype
        # Create dummy attention (Fully Connected / Uniform)
        # This approximates a "Global Smoothness" calculation
        A_in = torch.tensor(current_attn_matrix, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        A_sym = graph_engine.symmetrize_attention(A_in)
        A_adj = A_sym.squeeze(1)
        L_batch = graph_engine.construct_laplacian(A_adj)
        L = L_batch.squeeze(0)
        X = hidden_states.squeeze(0)
        energy = torch.trace(X.T @ L @ X)
        grad = torch.autograd.grad(energy, hidden_states)[0]
    return -grad # This is the "Smoothing" direction. 
                 # We will multiply by NEGATIVE strength to get "Sharpening".

def main():
    if not os.path.exists('output'): os.makedirs('output')
    
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager"
    )

    with open(DATA_FILE, 'r') as f:
        full_dataset = json.load(f)
        
    # Use Full N=300 for reliability
    eval_set = full_dataset 
    print(f"Evaluating on {len(eval_set)} samples.")

    # Spectral trust engine
    config = GSPConfig(normalization="none", symmetrization="symmetric")
    graph_engine = GraphConstructor(config)

    results = {}
    
    # 1. Establish Baseline (Again, for local consistency)
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

    # 2. Run Sweep
    for layer in LAYERS:
        condition_name = f"L{layer}_Sharpen"
        print(f"\n--- Running {condition_name} (Strength {STRENGTH}) ---")
        
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            
            # Dimensions
            if len(hidden.shape) == 3:
                seq_len = hidden.shape[1]
            else:
                seq_len = hidden.shape[0]
            
            # Dummy Attn
            dummy = np.ones((seq_len, seq_len)) / seq_len
            
            # Get Smoothing Gradient
            grad = get_spectral_gradient(hidden, dummy, graph_engine)
            
            # Normalize and Scale (Match experiments/19 logic)
            grad_norm = torch.norm(grad)
            if grad_norm > 1e-6:
                # STRENGTH is -1.0 (Sharpening)
                # We want perturbation magnitude to be 0.1 * hidden_norm
                target_mag = torch.norm(hidden) * 0.1 * abs(STRENGTH)
                perturbation = (grad / grad_norm) * target_mag * np.sign(STRENGTH)
            else:
                perturbation = torch.zeros_like(hidden)
            
            return (hidden + perturbation,) + output[1:] if isinstance(output, tuple) else (hidden + perturbation)
            
        handle = model.model.layers[layer].register_forward_hook(hook)
        
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
        
        # Immediate Save
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2)

    print("\nSweep Complete.")

if __name__ == "__main__":
    main()
