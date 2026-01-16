import torch
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os

# CONFIG
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
# OPTIMIZED PARAMETERS FROM PHASE 12
TARGET_LAYER = 12
STEER_STRENGTH = -0.01 # The "Micro-Dose"

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
    return -grad

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=TARGET_LAYER)
    parser.add_argument("--strength", type=float, default=STEER_STRENGTH)
    args = parser.parse_args()

    print(f"RUNNING PROOF OF LIFE ON {MODEL_NAME} @ L{args.layer}")
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
                
                # 1. Calculate Spec Vector
                spec_vec = get_smoothness_grad(hidden)
                spec_norm = torch.norm(spec_vec)
                
                perturbation = torch.zeros_like(hidden)
                
                # SHARPENING if strength is negative.
                # Logic: perturbation = vec * strength.
                # If vec is Smoothing direction, and strength is -0.01,
                # then perturbation moves AWAY from Smoothness (Sharpening).
                
                if condition == "spectral_steer":
                    if spec_norm > 1e-6:
                        # Normalize and Scale (0.1 * hidden_norm * strength)
                        # We use ABS(strength) for magnitude scalar, and the sign comes from strength itself.
                        # Wait, previous script logic:
                        # perturbation = spec_vec / spec_norm * torch.norm(hidden) * 0.1 * args.strength
                        # If strength is -0.01, perturbation is -0.001 * spec_vec. Correct.
                        perturbation = spec_vec / spec_norm * torch.norm(hidden) * 0.1 * args.strength
                
                elif condition == "control_random":
                    # STRICT NORM MATCHING
                    noise = torch.randn_like(hidden)
                    # We want the noise to have the SAME NORM as the spectral perturbation would have had.
                    # Target Norm = hidden_norm * 0.1 * abs(strength)
                    target_norm = torch.norm(hidden) * 0.1 * abs(args.strength)
                    perturbation = noise / torch.norm(noise) * target_norm
                    # Note: We don't apply sign to noise, just magnitude. Noise is random direction.
                
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
                gen_ids = out[0][input_len:]
                gen = tokenizer.decode(gen_ids, skip_special_tokens=True).lower()
                
                target = item.get('target_completion', item.get('target_word', 'wug')).lower()
                
                if target in gen:
                    success_count += 1
                elif "negation_trap" in item['type'] and "not" in gen:
                     success_count += 1
            except Exception as e:
                print(e)
        
        acc = success_count / len(data)
        results[condition] = acc
        print(f"Result {condition}: {acc:.2%}")
        
        if handle: handle.remove()

    if not os.path.exists('output'): os.makedirs('output')
    with open("output/llama_proof_of_life.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
