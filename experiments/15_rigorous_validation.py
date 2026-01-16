import torch
import json
import os
import numpy as np
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from spectral_trust import GSPConfig, GraphConstructor
from tqdm import tqdm

# CONFIG
MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
OUTPUT_FILE = "output/rigorous_results_N300.json"
DATA_FILE = "data/logic_dataset.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_LAYER = 24
TARGET_STRENGTH = 1.5

def get_spectral_gradient(hidden_states, current_attn_matrix, graph_engine):
    hidden_states = hidden_states.detach().requires_grad_(True)
    with torch.enable_grad():
        device = hidden_states.device
        dtype = hidden_states.dtype
        A_in = torch.tensor(current_attn_matrix, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        A_sym = graph_engine.symmetrize_attention(A_in)
        l_batch = graph_engine.construct_laplacian(A_sym.squeeze(1))
        L = l_batch.squeeze(0)
        X = hidden_states.squeeze(0)
        energy = torch.trace(X.T @ L @ X)
        grad = torch.autograd.grad(energy, hidden_states)[0]
    return -grad

def check_success(prompt_type, text, target):
    text_lower = text.lower()
    target_lower = target.lower() if target else ""
    if prompt_type == "negation_trap":
        return target_lower in text_lower
    elif prompt_type == "transitive_chain":
        return any(x in text_lower for x in ["contradiction", "impossible", "cannot be", "false"])
    return False

def run_condition(condition_name, model, tokenizer, eval_set, hook_fn=None):
    print(f"Running Condition: {condition_name}")
    handle = None
    if hook_fn:
        handle = model.model.layers[TARGET_LAYER].register_forward_hook(hook_fn)
    
    results = {}
    success_count = 0
    
    for item in tqdm(eval_set, desc=condition_name):
        inputs = tokenizer(item['prompt'], return_tensors="pt").to(DEVICE)
        try:
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
            gen_text = tokenizer.decode(out[0], skip_special_tokens=True)
            completion = gen_text[len(item['prompt']):]
            
            is_success = check_success(item['type'], completion, item.get('target_completion'))
            results[item['id']] = is_success
            if is_success: success_count += 1
                
        except Exception as e:
            print(f"Error on {item['id']}: {e}")
            results[item['id']] = False
            
    if handle:
        handle.remove()
        
    acc = success_count / len(eval_set)
    print(f"{condition_name} Accuracy: {acc:.2%}")
    return results, acc

def main():
    if not os.path.exists('output'): os.makedirs('output')
    
    # Load Data
    with open(DATA_FILE, 'r') as f:
        dataset = json.load(f)
    eval_set = [d for d in dataset if d['type'] in ["negation_trap", "transitive_chain"]]
    print(f"Loaded {len(eval_set)} samples.")

    # Load Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager"
    )
    
    # Engines
    config = GSPConfig(normalization="none", symmetrization="symmetric")
    graph_engine = GraphConstructor(config)

    # Defined Hooks
    def spectral_hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        seq_len = h.shape[1] if len(h.shape)==3 else h.shape[0]
        # Real Gradient
        dummy = np.ones((seq_len, seq_len))/seq_len
        grad = get_spectral_gradient(h, dummy, graph_engine)
        return (h + (TARGET_STRENGTH * grad),) + output[1:] if isinstance(output, tuple) else (h + (TARGET_STRENGTH * grad))

    def random_hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        seq_len = h.shape[1] if len(h.shape)==3 else h.shape[0]
        
        # 1. Compute Real Gradient to get the Norm target
        dummy = np.ones((seq_len, seq_len))/seq_len
        real_grad = get_spectral_gradient(h, dummy, graph_engine)
        grad_norm = torch.norm(real_grad)
        
        # 2. Generate Random Direction
        rand_vec = torch.randn_like(h)
        rand_norm = torch.norm(rand_vec)
        
        # 3. Normalize and Scale to match Real Gradient Norm
        if rand_norm > 1e-6:
            final_rand = (rand_vec / rand_norm) * grad_norm
        else:
            final_rand = torch.zeros_like(h)
            
        return (h + (TARGET_STRENGTH * final_rand),) + output[1:] if isinstance(output, tuple) else (h + (TARGET_STRENGTH * final_rand))

    def anti_hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        seq_len = h.shape[1] if len(h.shape)==3 else h.shape[0]
        dummy = np.ones((seq_len, seq_len))/seq_len
        grad = get_spectral_gradient(h, dummy, graph_engine)
        # NEGATIVE STRENGTH (Minimize Connectivity)
        return (h - (TARGET_STRENGTH * grad),) + output[1:] if isinstance(output, tuple) else (h - (TARGET_STRENGTH * grad))

    # Run Conditions
    final_output = {"N": len(eval_set)}
    
    # 1. Baseline
    res_base, acc_base = run_condition("Baseline", model, tokenizer, eval_set, hook_fn=None)
    final_output["Baseline"] = {"accuracy": acc_base, "results": res_base}
    
    # 2. Optimal (Spectral)
    res_spec, acc_spec = run_condition(f"Spectral_L{TARGET_LAYER}", model, tokenizer, eval_set, hook_fn=spectral_hook)
    final_output["Spectral"] = {"accuracy": acc_spec, "results": res_spec}
    
    # 3. Control A (Random)
    res_rand, acc_rand = run_condition("Control_Random", model, tokenizer, eval_set, hook_fn=random_hook)
    final_output["Control_Random"] = {"accuracy": acc_rand, "results": res_rand}
    
    # 4. Control B (Anti)
    res_anti, acc_anti = run_condition("Control_Anti", model, tokenizer, eval_set, hook_fn=anti_hook)
    final_output["Control_Anti"] = {"accuracy": acc_anti, "results": res_anti}

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_output, f, indent=2)
    print("Rigorous validation complete.")

if __name__ == "__main__":
    main()
