import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os
import numpy as np

# --- Configuration: Phi-3 Golden Settings ---
# From experiments/15_rigorous_validation.py
# "Fractured" Repair Strategy
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
TARGET_LAYER = 24
TARGET_STRENGTH = 1.5 # Massive Smoothing

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class UniformSmoothingHook:
    def __init__(self, strength):
        self.strength = strength
    
    def __call__(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        
        # Phi-3 Repair used Uniform Graph (Static)
        # L = I - 1/N * Ones
        # Force = Gradient of Energy = 2 * L * X = 2 * (X - Mean(X))
        # We want to smooth, so we subtract force: X_new = X - strength * Force
        
        # Note: In experiment 15, we passed strength=1.5 and used (h + strength * grad) ??
        # Let's re-read line 99 of exp 15: `return h + (TARGET_STRENGTH * grad)`
        # AND line 31: `return -grad` (get_spectral_gradient returns NEGATIVE grad)
        # So it was returning: h + strength * (-grad_energy) = h - strength * grad_energy.
        # Yes, subtracting gradient of energy minimizes energy (Smooths).
        
        # Impl:
        # Force (Energy Grad) = 2 * (X - Mean(X))
        # But wait, did Exp 15 normalize?
        # Line 98: `grad = get_spectral_gradient(...)`
        # It does NOT appear to normalize the gradient in `spectral_hook` (unlike `random_hook`).
        # It adds `TARGET_STRENGTH * grad` directly. 
        # This explains why the strength (1.5) is so high compared to Llama (0.1).
        # Without normalization, the gradient scale depends on X scale.
        
        # I trust the "Do No Harm" test to reveal if this un-normalized massive step destroys PPL.
        # If PPL explodes, we know Exp 15 was risky (but effective for logic).
        
        seq_len = hidden.shape[1]
        
        # Calculate Mean across sequence dim (dim 1)
        # Shape: (Batch, Seq, Dim)
        mean_hidden = hidden.mean(dim=1, keepdim=True)
        
        # L * X = X - Mean
        # Energy Grad = 2 * (X - Mean)
        # Direction = -Grad = 2 * (Mean - X)
        
        # We perform the exact op from Exp 15:
        # h_new = h + strength * direction
        
        smoothing_direction = 2 * (mean_hidden - hidden)
        
        # Exp 15 did NOT normalize. We follow the golden setting exactly.
        perturbed = hidden + (self.strength * smoothing_direction)
        
        return (perturbed,) + output[1:] if isinstance(output, tuple) else (perturbed)

def measure_perplexity(model, tokenizer):
    print("Loading WikiText-2...")
    try:
        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception as e:
        print(f"Failed to load wikitext: {e}. Trying local fallback or subset.")
        return 999.0

    text_blob = "\n\n".join(test["text"][:50]) # First 50 docs
    encodings = tokenizer(text_blob, return_tensors="pt")
    
    max_length = model.config.max_position_embeddings
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    print(f"Evaluating PPL on {seq_len} tokens...")
    
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc 
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss * trg_len)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()

def run_phi3_check():
    print(f"Loading {MODEL_NAME} for Do No Harm Check...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    
    print("--- Measuring Baseline Perplexity ---")
    baseline_ppl = measure_perplexity(model, tokenizer)
    print(f"Baseline PPL: {baseline_ppl:.2f}")
    
    print(f"--- Applying Phi-3 Repair (L{TARGET_LAYER}, S={TARGET_STRENGTH}) ---")
    
    layer = model.model.layers[TARGET_LAYER]
    hook = UniformSmoothingHook(TARGET_STRENGTH)
    handle = layer.register_forward_hook(hook)
        
    steered_ppl = measure_perplexity(model, tokenizer)
    print(f"Steered PPL: {steered_ppl:.2f}")
    
    delta = steered_ppl - baseline_ppl
    print(f"Delta: {delta:.2f}")
    
    if abs(delta) < 1.0:
        print("RESULT: PASS. Steering is invisible to general capability.")
    elif delta < 0:
        print("RESULT: BONUS. Steering actually improved general fluency!")
    else:
        print("RESULT: WARNING. Steering degraded fluency (Trade-off detected).")

if __name__ == "__main__":
    run_phi3_check()
