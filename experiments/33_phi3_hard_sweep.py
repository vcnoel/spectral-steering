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
ALPHAS = [0.2, 0.3, 0.4] # The Goldilocks Zone
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOGIC_DATA = "data/logic_dataset.json"

class UniformSmoothingHook:
    def __init__(self, alpha):
        self.alpha = alpha
    
    def __call__(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        
        # Normalized Uniform Smoothing
        mean_hidden = hidden.mean(dim=1, keepdim=True)
        smoothing_dir = 2 * (mean_hidden - hidden)
        
        dir_norm = torch.norm(smoothing_dir, p=2, dim=-1, keepdim=True)
        hidden_norm = torch.norm(hidden, p=2, dim=-1, keepdim=True)
        dir_norm = torch.clamp(dir_norm, min=1e-6)
        
        # Perturbation
        perturbation = (smoothing_dir / dir_norm) * hidden_norm * self.alpha
        
        return (hidden + perturbation,) + output[1:] if isinstance(output, tuple) else (hidden + perturbation)

def measure_perplexity(model, tokenizer):
    try:
        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text_blob = "\n\n".join(test["text"][:30]) # Speed mode
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
    # Evaluating on full set of Chains (likely N=100-150)
    
    for item in tqdm(eval_set, desc="Logic Eval"):
        inputs = tokenizer(item['prompt'], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen_text = tokenizer.decode(out[0], skip_special_tokens=True).lower()
        
        # Check Transitive Chain Failure indicators
        # Success if it detects "contradiction", "impossible", "cannot be", "false"
        success = any(x in gen_text for x in ["contradiction", "impossible", "cannot be", "false", "no,"])
        if success: success_count += 1
            
    return success_count / len(eval_set)

def main():
    box_print(f"Phi-3 Hard Task Sweep (Transitive Rings)")
    
    # 1. Load Resources
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    
    with open(LOGIC_DATA, 'r') as f:
        full_data = json.load(f)
    # Filter for Transitive Chains (Hard Task)
    eval_set = [d for d in full_data if d['type'] == "transitive_chain"]
    print(f"Logic Eval Subset: {len(eval_set)} Transitive Chains.");

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
        
        # Only eval logic if PPL is "Safe" (< 20)
        if ppl < 20.0:
            logic_acc = check_logic_accuracy(model, tokenizer, eval_set)
        else:
            print(f"Skipping Logic Eval (PPL {ppl:.2f} > 20)")
            logic_acc = 0.0
            
        handle.remove()
        
        results[f"S{alpha}"] = {"PPL": ppl, "Logic": logic_acc}
        print(f"Alpha {alpha} -> PPL: {ppl:.2f}, Logic: {logic_acc:.2%}")
        
        with open("output/phi3_hard_sweep.json", 'w') as f:
            json.dump(results, f, indent=2)

    print("\nHard Sweep Complete.")

if __name__ == "__main__":
    main()
