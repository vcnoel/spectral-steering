import torch
import json
import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# CONFIG
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
TARGET_LAYER = 24
ALPHAS = [0.1, 0.3, 0.5, 0.8, 1.0] # Normalized Strengths
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOGIC_DATA = "data/logic_dataset.json"

class UniformSmoothingHook:
    def __init__(self, alpha):
        self.alpha = alpha
    
    def __call__(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        
        # Uniform Smoothing Direction calculation
        # L = I - 1/N * Ones
        # Energy Grad = 2 * (X - Mean)
        # Smoothing Direction = -Grad = 2 * (Mean - X)
        
        mean_hidden = hidden.mean(dim=1, keepdim=True)
        smoothing_dir = 2 * (mean_hidden - hidden)
        
        # RELATIVE NORM SCALING (The Fix)
        # We normalize the direction vector, then scale it by the hidden state norm * alpha.
        # This ensures alpha represents a percentage of the signal energy.
        
        dir_norm = torch.norm(smoothing_dir, p=2, dim=-1, keepdim=True)
        hidden_norm = torch.norm(hidden, p=2, dim=-1, keepdim=True)
        
        # Avoid div by zero
        dir_norm = torch.clamp(dir_norm, min=1e-6)
        
        # Perturbation = (Dir / |Dir|) * |Hidden| * Alpha
        perturbation = (smoothing_dir / dir_norm) * hidden_norm * self.alpha
        
        return (hidden + perturbation,) + output[1:] if isinstance(output, tuple) else (hidden + perturbation)

def measure_perplexity(model, tokenizer):
    try:
        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text_blob = "\n\n".join(test["text"][:30]) # Reduced to 30 docs for speed in sweep
        encodings = tokenizer(text_blob, return_tensors="pt")
    except:
        return 999.0

    max_length = model.config.max_position_embeddings
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    
    for begin_loc in range(0, seq_len, stride):
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

def box_print(msg):
    print("+" + "-" * (len(msg) + 2) + "+")
    print("| " + msg + " |")
    print("+" + "-" * (len(msg) + 2) + "+")

def check_logic_accuracy(model, tokenizer, eval_set):
    success_count = 0
    # Fast check on subset (N=50) to save time, or full?
    # Let's do partial N=50 to be quick for sweep
    subset = eval_set[:50] 
    
    for item in subset:
        inputs = tokenizer(item['prompt'], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=15, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen_text = tokenizer.decode(out[0], skip_special_tokens=True).lower()
        
        target = item.get('target_completion', item.get('target_word', 'wug')).lower()
        if target in gen_text: success_count += 1
        elif "negation_trap" in item['type'] and "not" in gen_text: success_count += 1
            
    return success_count / len(subset)

def main():
    box_print(f"Phi-3 Rescue Mission: L{TARGET_LAYER} Safety Sweep")
    
    # 1. Load Resources
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    
    with open(LOGIC_DATA, 'r') as f:
        full_data = json.load(f)
    # Filter for Negation Trap (the hardest one for Phi-3)
    eval_set = [d for d in full_data if d['type'] == "negation_trap"]
    print(f"Logic Eval Subset: {len(eval_set)} Negation Traps (using 50).");

    results = {}

    # 2. Baseline
    print("\n--- Baseline ---")
    base_ppl = measure_perplexity(model, tokenizer)
    base_acc = check_logic_accuracy(model, tokenizer, eval_set)
    results["Baseline"] = {"PPL": base_ppl, "Logic": base_acc}
    print(f"Baseline -> PPL: {base_ppl:.2f}, Logic: {base_acc:.2%}")

    # 3. Sweep
    for alpha in ALPHAS:
        print(f"\n--- Testing Alpha {alpha} ---")
        
        hook = UniformSmoothingHook(alpha)
        handle = model.model.layers[TARGET_LAYER].register_forward_hook(hook)
        
        ppl = measure_perplexity(model, tokenizer)
        # Only check logic if PPL is sane (< 100) to save time?
        # Actually check both to see the trade-off curve
        if ppl < 1000:
            logic_acc = check_logic_accuracy(model, tokenizer, eval_set)
        else:
            logic_acc = 0.0 # Broken
            
        handle.remove()
        
        results[f"S{alpha}"] = {"PPL": ppl, "Logic": logic_acc}
        print(f"Alpha {alpha} -> PPL: {ppl:.2f}, Logic: {logic_acc:.2%}")
        
        # Save incremental
        with open("output/phi3_safety_sweep.json", 'w') as f:
            json.dump(results, f, indent=2)

    print("\nRescue Mission Complete.")

if __name__ == "__main__":
    main()
