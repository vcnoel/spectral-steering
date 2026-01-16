import torch
import json
import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from spectral_trust import GSPConfig, GraphConstructor
from tqdm import tqdm

# CONFIG
MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
OUTPUT_FILE = "output/logic_sweep_results.json"
DATA_FILE = "data/logic_dataset.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Sweep Parameters
LAYERS = [16, 18, 20, 22, 24]
STRENGTHS = [1.0, 1.5, 2.0]

# --- Helper Functions ---
def get_spectral_gradient(hidden_states, current_attn_matrix, graph_engine):
    hidden_states = hidden_states.detach().requires_grad_(True)
    with torch.enable_grad():
        device = hidden_states.device
        dtype = hidden_states.dtype
        A_in = torch.tensor(current_attn_matrix, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        A_sym = graph_engine.symmetrize_attention(A_in)
        A_adj = A_sym.squeeze(1)
        L_batch = graph_engine.construct_laplacian(A_adj)
        L = L_batch.squeeze(0)
        X = hidden_states.squeeze(0)
        energy = torch.trace(X.T @ L @ X)
        grad = torch.autograd.grad(energy, hidden_states)[0]
    return -grad

def check_success(prompt_type, text, target):
    text_lower = text.lower()
    target_lower = target.lower() if target else ""
    
    if prompt_type == "negation_trap":
        # Success if it correctly concludes "not a [target]"
        # Failure if it says "not a cat" when it should say "not a dog"? 
        # Wait, the prompt is "This object is NOT {adj}". Target is {cat}. 
        # "Therefore this object is definitely not a {cat}".
        # So we check if {target} is in the output.
        return target_lower in text_lower
    
    elif prompt_type == "transitive_chain":
        # Success if it mentions contradiction or impossibility
        return any(x in text_lower for x in ["contradiction", "impossible", "cannot be", "false"])
    
    return False # Nested clauses hard to auto-verify without grammar parser

def main():
    if not os.path.exists('output'): os.makedirs('output')
    
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager"
    )

    with open(DATA_FILE, 'r') as f:
        dataset = json.load(f)
        
    # Filter for auto-checkable types
    eval_set = [d for d in dataset if d['type'] in ["negation_trap", "transitive_chain"]]
    print(f"Evaluating on {len(eval_set)} samples (Negation & Chains).")

    # Spectral Trust Engine
    config = GSPConfig(normalization="none", symmetrization="symmetric")
    graph_engine = GraphConstructor(config)

    results = {}

    # --- Baseline ---
    print("Running Baseline...")
    success_count = 0
    for item in tqdm(eval_set):
        inputs = tokenizer(item['prompt'], return_tensors="pt").to(DEVICE)
        
        # Phi-3 is chatty, limit tokens
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        gen_text = tokenizer.decode(out[0], skip_special_tokens=True)
        # Extract only the completion
        completion = gen_text[len(item['prompt']):]
        
        if check_success(item['type'], completion, item.get('target_completion')):
            success_count += 1
            
    base_acc = success_count / len(eval_set)
    results["baseline"] = base_acc
    print(f"Baseline Accuracy: {base_acc:.2%}")

    # --- Sweep ---
    for layer in LAYERS:
        for strength in STRENGTHS:
            config_name = f"L{layer}_S{strength}"
            print(f"Running {config_name}...")
            
            # Define Hook
            def hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                # Check shape
                if len(h.shape) == 3: # [Batch, Seq, Dim]
                    seq_len = h.shape[1]
                else:
                    seq_len = h.shape[0]
                    
                dummy_attn = np.ones((seq_len, seq_len)) / seq_len
                grad = get_spectral_gradient(h, dummy_attn, graph_engine)
                
                new_h = h + (strength * grad)
                return (new_h,) + output[1:] if isinstance(output, tuple) else new_h

            # Register
            handle = model.model.layers[layer].register_forward_hook(hook)
            
            success_count = 0
            for item in tqdm(eval_set):
                inputs = tokenizer(item['prompt'], return_tensors="pt").to(DEVICE)
                try:
                    out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
                    gen_text = tokenizer.decode(out[0], skip_special_tokens=True)
                    completion = gen_text[len(item['prompt']):]
                    
                    if check_success(item['type'], completion, item.get('target_completion')):
                        success_count += 1
                except Exception as e:
                    print(f"Error: {e}")
            
            handle.remove()
            
            acc = success_count / len(eval_set)
            results[config_name] = acc
            print(f"{config_name} Accuracy: {acc:.2%}")
            
            # Save intermediate
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(results, f, indent=2)

    print("Sweep Complete.")

if __name__ == "__main__":
    main()
