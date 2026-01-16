import torch
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os

# CONFIG
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# TARGET_LAYER updated after sensitivity scan (L7 was most brittle @ 22%)
TARGET_LAYER = 7  
STEER_STRENGTH = 1.5

def get_smoothness_grad(hidden):
    """Minimizes Dirichlet Energy (Maximizes Smoothness)"""
    hidden = hidden.detach().requires_grad_(True)
    
    with torch.enable_grad():
        # Cast to float32 for stability
        X = hidden.squeeze(0).float()
        # Approx Graph: Gram matrix
        # Normalize by sqrt(dim)
        A = torch.softmax((X @ X.T) / (X.shape[-1]**0.5), dim=-1)
        D = torch.diag(A.sum(1))
        L = D - A
        energy = torch.trace(X.T @ L @ X)
        grad = torch.autograd.grad(energy, hidden)[0]
        
    return -grad

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=TARGET_LAYER)
    parser.add_argument("--strength", type=float, default=STEER_STRENGTH)
    args = parser.parse_args()

    print(f"RUNNING RIGOROUS VALIDATION ON {MODEL_NAME} @ L{args.layer}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

    with open('data/logic_dataset.json', 'r') as f:
        data = json.load(f) # Full N=300

    results = {
        "baseline": 0,
        "control_random": 0,
        "spectral_steer": 0
    }

    # CONDITIONS LOOP
    for condition in ["baseline", "control_random", "spectral_steer"]:
        print(f"\n--- Testing Condition: {condition} ---")
        
        handle = None
        if condition != "baseline":
            def hook(module, input_args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                
                # 1. Calculate Spectral Vector (The Signal)
                spec_vec = get_smoothness_grad(hidden)
                spec_norm = torch.norm(spec_vec)
                
                perturbation = torch.zeros_like(hidden)
                
                if condition == "spectral_steer":
                    # Normalize to relative scale
                    if spec_norm > 1e-6:
                        perturbation = spec_vec / spec_norm * torch.norm(hidden) * 0.1 * args.strength
                    else:
                        perturbation = torch.zeros_like(hidden)
                
                elif condition == "control_random":
                    # STRICT NORM MATCHING
                    noise = torch.randn_like(hidden)
                    # Scale noise to match the EXACT norm of the spectral vector we would have used
                    # But wait, we want to match the INTENDED perturbation norm
                    target_norm = torch.norm(hidden) * 0.1 * args.strength
                    perturbation = noise / torch.norm(noise) * target_norm
                
                return (hidden + perturbation,) + output[1:] if isinstance(output, tuple) else (hidden + perturbation)

            handle = model.model.layers[args.layer].register_forward_hook(hook)

        # EVAL LOOP
        success_count = 0
        for item in tqdm(data):
            prompt = item['prompt']
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            try:
                input_len = inputs.input_ids.shape[1]
                out = model.generate(**inputs, max_new_tokens=15, pad_token_id=tokenizer.eos_token_id)
                # Slice logic to remove prompt
                gen_ids = out[0][input_len:]
                gen = tokenizer.decode(gen_ids, skip_special_tokens=True).lower()
                
                target = item.get('target_completion', item.get('target_word', 'wug')).lower()
                
                # Check for success
                if target in gen:
                    success_count += 1
                elif "negation_trap" in item['type'] and "not" in gen:
                     # e.g. "not a wug" if that is considered correct for negation trap logic in some contexts, 
                     # but strictly we usually want the target word alone or specific structure. 
                     # sticking to user's simple heuristic: target OR "not" (which is generous for negations but standard for this sensitivity check)
                     success_count += 1
            except Exception as e:
                print(e)
        
        acc = success_count / len(data)
        results[condition] = acc
        print(f"Result {condition}: {acc:.2%}")
        
        if handle: handle.remove()

    if not os.path.exists('output'): os.makedirs('output')
    with open("output/qwen_rigorous_results.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
