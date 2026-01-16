import torch
import numpy as np
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_frequency_noise(shape, freq_range, device='cuda'):
    """
    Generates noise concentrated in a specific frequency band using FFT.
    freq_range: tuple (low_frac, high_frac) between 0.0 and 1.0
    """
    # 1. Create White Noise
    white_noise = torch.randn(shape, device=device)
    
    # 2. FFT (along the embedding dimension)
    fft_noise = torch.fft.rfft(white_noise, dim=-1)
    
    # 3. Mask Frequencies
    mask = torch.zeros_like(fft_noise, device=device)
    seq_len = fft_noise.shape[-1] # This is actually (D // 2) + 1
    
    # Calculate indices corresponding to freq_range (0.0 to 1.0)
    # Note: dim=-1 is embedding dimension D.
    start_idx = int(freq_range[0] * seq_len)
    end_idx = int(freq_range[1] * seq_len)
    
    # Ensure safe indices
    start_idx = max(0, start_idx)
    end_idx = min(seq_len, end_idx)
    
    mask[:, :, start_idx:end_idx] = 1.0
    
    # 4. Inverse FFT
    filtered_noise = torch.fft.irfft(fft_noise * mask, n=shape[-1], dim=-1)
    
    # Normalize energy to unit norm per vector, then scaled later
    # Prevent division by zero
    filtered_noise = filtered_noise / (filtered_noise.norm(dim=-1, keepdim=True) + 1e-8)
    
    return filtered_noise

def spectral_stress_test(model, tokenizer, prompt, target_layer_idx, device=None):
    """
    Shakes the model at Low, Mid, and High frequencies and measures output change.
    """
    if device is None:
        device = model.device
        
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print("Computing Baseline Logits...")
    with torch.no_grad():
        base_logits = model(**inputs).logits
    base_probs = torch.softmax(base_logits, dim=-1)
    
    # Frequency Bands
    bands = {
        "Low Freq (Smoothing)": (0.0, 0.1),   # The "Bass" - Structure
        "Mid Freq (Content)":   (0.1, 0.5),   # The "Vocals" - Semantics
        "High Freq (Edges)":    (0.5, 1.0)    # The "Treble" - Texture/Detail
    }
    
    results = {}
    
    print(f"\n--- Stress Testing Layer {target_layer_idx} ---")
    print(f"Prompt: '{prompt}'")
    
    for band_name, freq_range in bands.items():
        # Define Hook
        def hook(module, input, output):
            # output is tuple (hidden_states, ...)
            hidden = output[0] if isinstance(output, tuple) else output
            
            # Generate Band-Limited Noise
            noise = generate_frequency_noise(hidden.shape, freq_range, device)
            
            # Scale to 10% of mean norm (safer/standard)
            scale = 0.1 * hidden.norm(dim=-1, keepdim=True).mean() 
            
            perturbation = (noise * scale).to(hidden.dtype)
            perturbed = hidden + perturbation
            
            return (perturbed,) + output[1:] if isinstance(output, tuple) else perturbed

        # Register Hook
        layer = model.model.layers[target_layer_idx]
        handle = layer.register_forward_hook(hook)
        
        # Forward Pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        handle.remove()
        
        # Measure Deviation (KL Divergence from Baseline)
        # Cast to float32 for stability
        new_logits = outputs.logits.float()
        base_logits_f32 = base_logits.float()
        
        new_probs = torch.softmax(new_logits, dim=-1)
        base_probs_f32 = torch.softmax(base_logits_f32, dim=-1)
        
        # KL(P || Q) = sum(P * log(P/Q))
        eps = 1e-6
        kl_div = torch.sum(base_probs_f32 * (torch.log(base_probs_f32 + eps) - torch.log(new_probs + eps)), dim=-1).mean()
        
        val = kl_div.item()
        results[band_name] = val
        print(f"{band_name:<20} | KL Divergence: {val:.4f}")

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--layer", type=int, default=2)
    args = parser.parse_args()
    
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
    
    # A standard prompt to test structure vs texture
    prompt = "The concept of logic is distinct from distinctness because"
    
    spectral_stress_test(model, tokenizer, prompt, args.layer)

if __name__ == "__main__":
    main()
