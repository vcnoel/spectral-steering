import torch
import json
import argparse
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os

# CONFIG
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
TARGET_LAYER = 2 
DEFAULT_STRENGTH = 1.0

# --- BandPass Entropy Loss (The "Precision Medicine") ---
class BandPassEntropyLoss(nn.Module):
    def __init__(self, target_band=(0.1, 1.0), epsilon=1e-8):
        """
        Maximizes information density (Entropy) specifically in the target frequency band.
        Targeting Mid-High Frequencies for Llama 3.2.
        """
        super().__init__()
        self.low_f = target_band[0]
        self.high_f = target_band[1]
        self.epsilon = epsilon

    def forward(self, hidden_states):
        """
        Input: Hidden states [Batch, Seq_Len, Dim]
        Output: Scalar Loss (Negative Entropy -> Minimizing this Maximizes Entropy)
        """
        # FFT along feature dimension (Dim)
        # Cast to float32 for FFT stability/support
        fft_out = torch.fft.rfft(hidden_states.float(), dim=-1)
        
        # Power Spectrum
        power_spectrum = fft_out.abs().pow(2)
        
        # Band-Pass Filter
        n_freqs = power_spectrum.shape[-1]
        start_idx = int(self.low_f * n_freqs)
        end_idx = int(self.high_f * n_freqs)
        
        # Ensure indices result in non-empty band
        if start_idx >= end_idx:
            start_idx = 0
            end_idx = n_freqs
            
        band_power = power_spectrum[..., start_idx:end_idx]
        
        # Normalize to PDF
        total_band_energy = band_power.sum(dim=-1, keepdim=True) + self.epsilon
        pdf = band_power / total_band_energy
        
        # Entropy H = -Sum(p * log(p))
        entropy = -(pdf * torch.log(pdf + self.epsilon)).sum(dim=-1)
        
        # Return Negative Entropy (to minimize)
        return -entropy.mean()

def get_entropy_grad(hidden, criterion):
    """Calculates gradient of BandPassEntropyLoss w.r.t hidden states"""
    hidden = hidden.detach().requires_grad_(True)
    with torch.enable_grad():
        loss = criterion(hidden)
        grad = torch.autograd.grad(loss, hidden)[0]
    return grad # We return +grad because 'loss' is Negative Entropy. 
                # Minimizing Loss -> Moving against Gradient steps? 
                # Wait. Standard Optimization: w = w - lr * grad. 
                # If we use `hidden + (-grad * strength)`, we are doing gradient descent on Loss.
                # Gradient Descent on (-Entropy) => Maximizing Entropy.
                # So we return +grad, and subtract it later (or return -grad and add it).
                # Let's match previous scripts: return -grad.
    return -grad

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=TARGET_LAYER)
    parser.add_argument("--strength", type=float, default=DEFAULT_STRENGTH)
    args = parser.parse_args()

    print(f"RUNNING ENTROPY STEERING ON {MODEL_NAME} @ L{args.layer}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

    with open('data/logic_dataset.json', 'r') as f:
        data = json.load(f) # Full N=300

    results = {
        "baseline": 0,
        "entropy_steer": 0
    }
    
    # Initialize Loss Module
    # Target Band 0.1-1.0 (Mid-High)
    entropy_criterion = BandPassEntropyLoss(target_band=(0.1, 1.0))

    # CONDITIONS LOOP
    for condition in ["baseline", "entropy_steer"]:
        print(f"\n--- Testing Condition: {condition} ---")
        
        handle = None
        if condition == "entropy_steer":
            def hook(module, input_args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                
                # Get Gradient of Negative Entropy
                # grad points in direction of INCREASING Loss (-Entropy) => DECREASING Entropy
                # We want to DECREASE Loss => MAXIMIZE Entropy
                # So we should move OPPOSITE to grad.
                # get_entropy_grad returns -grad. 
                # So adding (strength * -grad) is Gradient Descent.
                
                neg_grad = get_entropy_grad(hidden, entropy_criterion)
                
                # Normalize Gradient for stability (trust region)
                grad_norm = torch.norm(neg_grad)
                if grad_norm > 1e-6:
                     # Scale similarly to previous experiments: 0.1 * hidden_norm
                     target_norm = 0.1 * torch.norm(hidden) * args.strength
                     perturbation = (neg_grad / grad_norm) * target_norm
                else:
                    perturbation = torch.zeros_like(hidden)

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
    with open("output/llama_entropy_results.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
