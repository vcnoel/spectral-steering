import torch
import json
import os
import re
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# CONFIG
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
TARGET_LAYER = 24
ALPHAS = [0.2, 0.3, 0.4]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
        text_blob = "\n\n".join(test["text"][:30])
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

def extract_answer(text):
    # GSM8K usually has #### [Number] at end
    # Or strict format.
    # We will look for the last number
    try:
        # Look for "#### [number]"
        matches = re.findall(r"####\s*(-?\d+\.?\d*)", text)
        if matches: return float(matches[-1])
        
        # Fallback: Look for last number
        # matches = re.findall(r"-?\d+\.?\d*", text)
        # if matches: return float(matches[-1])
    except:
        pass
    return None

def eval_gsm8k(model, tokenizer, dataset):
    correct = 0
    total = 0
    
    # N=20 Subset for speed
    for i, item in enumerate(tqdm(dataset)):
        if i >= 20: break
        
        prompt = f"Question: {item['question']}\nLet's think step by step.\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        
        gen_text = tokenizer.decode(out[0], skip_special_tokens=True)
        # Extract answer from generation
        # Phi-3 usually follows "#### Answer" if trained on it, but we might need 
        # to parse the reasoning.
        # Actually, let's just check for the target answer logic used in typical eval scripts.
        # Simple string match of the target number?
        
        target_str = item['answer'].split("####")[-1].strip()
        try:
            target_val = float(target_str.replace(',', ''))
        except:
             continue
             
        # Check if target val is in output
        # This is a loose heuristic but standard for quick checking
        if str(target_str) in gen_text:
            correct += 1
            
        total += 1
        
    return correct / total

def main():
    print(f"Phi-3 GSM8K Sweep (Values Add)")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    
    print("Loading GSM8K...")
    gsm8k = load_dataset("gsm8k", "main", split="test")
    print(f"Loaded {len(gsm8k)} samples.")
    
    # Negative Alphas (Sharpening) to see if Math improves
    alphas = [-0.1, -0.2]
    
    results = {}

    # 1. Baseline
    print("\n--- Baseline ---")
    base_ppl = measure_perplexity(model, tokenizer)
    base_acc = eval_gsm8k(model, tokenizer, gsm8k)
    results["Baseline"] = {"PPL": base_ppl, "Score": base_acc}
    print(f"Baseline -> PPL: {base_ppl:.2f}, GSM8K: {base_acc:.2%}")

    # 2. Sweep
    for alpha in alphas:
        print(f"\n--- Testing Alpha {alpha} ---")
        
        hook = UniformSmoothingHook(alpha)
        handle = model.model.layers[TARGET_LAYER].register_forward_hook(hook)
        
        ppl = measure_perplexity(model, tokenizer)
        
        if ppl < 20.0:
            acc = eval_gsm8k(model, tokenizer, gsm8k)
        else:
            print(f"Skipping Eval (PPL {ppl:.2f} > 20)")
            acc = 0.0
            
        handle.remove()
        
        results[f"S{alpha}"] = {"PPL": ppl, "Score": acc}
        print(f"Alpha {alpha} -> PPL: {ppl:.2f}, GSM8K: {acc:.2%}")
        
        with open("output/phi3_gsm8k_sweep.json", 'w') as f:
            json.dump(results, f, indent=2)

    print("\nGSM8K Sweep Complete.")

if __name__ == "__main__":
    main()
