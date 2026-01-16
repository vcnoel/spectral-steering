import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
# Assuming spectral_trust is installed or available. 
# But for Qwen "Smoothness" we can implement locally as per user snippet.
# from spectral_trust import compute_laplacian, symmetrize_attention 

import argparse

# CONFIG
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

def get_smoothness_gradient(hidden_states):
    """
    Returns gradient to MINIMIZE Dirichlet Energy (Maximize Smoothness).
    Assumes latent graph structure is approximated by hidden state similarity.
    """
    hidden_states = hidden_states.detach().requires_grad_(True)
    
    with torch.enable_grad():
        # Cast to float32 for stability and torch.trace support
        X = hidden_states.squeeze(0).float() # [Seq, Dim]
        gram = X @ X.T
        # Simple adjacency proxy
        # Normalize by sqrt(dim) for stability
        A = torch.softmax(gram / (X.shape[-1]**0.5), dim=-1)
        
        # Laplacian
        D = torch.diag(A.sum(dim=1))
        L = D - A
        
        # Energy = Trace(X.T @ L @ X)
        energy = torch.trace(X.T @ L @ X)
        
        # Gradient to MINIMIZE energy
        grad = torch.autograd.grad(energy, hidden_states)[0]
    return -grad

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=18, help="Target layer for steering")
    parser.add_argument("--strength", type=float, default=1.5, help="Steering strength")
    args = parser.parse_args()

    TARGET_LAYER = args.layer
    STEERING_STRENGTH = args.strength

    print(f"Steering {MODEL_NAME} at Layer {TARGET_LAYER} with Strength {STEERING_STRENGTH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    
    # Load Data
    with open('data/logic_dataset.json', 'r') as f:
        data = json.load(f)

    # Steering Hook
    def steer_hook(module, hook_args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        grad = get_smoothness_gradient(hidden)
        
        # Normalize grad to avoid exploding values
        norm_grad = torch.norm(grad)
        norm_hidden = torch.norm(hidden)
        
        if norm_grad > 1e-6:
             grad = grad / norm_grad * norm_hidden * 0.1 # Relative scale 0.1 * Strength
        
        return (hidden + (STEERING_STRENGTH * grad),) + output[1:] if isinstance(output, tuple) else (hidden + (STEERING_STRENGTH * grad))

    # Register
    handle = model.model.layers[TARGET_LAYER].register_forward_hook(steer_hook)
    
    # Run Validation (N=50)
    success = 0
    total = 50
    for i in range(total):
        item = data[i]
        prompt = item['prompt']
        
        messages = [{"role": "user", "content": prompt}]
        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
        
        input_len = inputs.input_ids.shape[1]
        out = model.generate(**inputs, max_new_tokens=15, pad_token_id=tokenizer.eos_token_id)
        
        # Slice output to remove prompt
        gen_ids = out[0][input_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).lower()
        
        target = item.get('target_completion', 'wug').lower()
        if target in text:
            success += 1
        elif item['type'] == 'transitive_chain' and any(x in text for x in ["contradiction", "impossible", "false"]):
            success += 1
            
    print(f"Steered Accuracy: {success/total:.2%}")

if __name__ == "__main__":
    main()
