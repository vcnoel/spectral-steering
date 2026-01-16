import torch
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os

# CONFIG
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_FILE = "output/qwen_full_scan.json"

def main():
    parser = argparse.ArgumentParser()
    # Default to 0.0 for calibration/baseline check
    parser.add_argument("--noise_scale", type=float, default=0.0)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct") 
    args = parser.parse_args()

    MODEL_NAME = args.model
    print(f"FULL SENSITIVITY SCAN: {MODEL_NAME} (Noise Scale: {args.noise_scale})")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    
    # Load 50 samples for speed (we just need the signal)
    with open('data/logic_dataset.json', 'r') as f:
        data = json.load(f)[:50] 

    results = {}
    num_layers = model.config.num_hidden_layers
    
    # SCAN EVERY LAYER
    for layer_idx in range(num_layers):
        
        # Noise Hook
        def noise_hook(module, input_args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            noise = torch.randn_like(hidden)
            # Strict Norm Matching
            noise = noise / torch.norm(noise, dim=-1, keepdim=True) * torch.norm(hidden, dim=-1, keepdim=True)
            
            return (hidden + (args.noise_scale * noise),) + output[1:] if isinstance(output, tuple) else (hidden + (args.noise_scale * noise))

        handle = model.model.layers[layer_idx].register_forward_hook(noise_hook)
        
        success = 0
        for i, item in enumerate(data):
            prompt = item['prompt']
            # Chat template for Qwen
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            try:
                input_len = inputs.input_ids.shape[1]
                out = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
                # Slice logic to remove prompt
                gen_ids = out[0][input_len:]
                gen = tokenizer.decode(gen_ids, skip_special_tokens=True).lower()
                
                # Check target (handle target_completion or target_word)
                target = item.get('target_completion', item.get('target_word', 'wug')).lower()
                
                is_success = False
                if target in gen or ("negation_trap" in item['type'] and "not" in gen):
                    is_success = True
                    success += 1
                
                # Debug print for first few failures or samples of Layer 0
                if layer_idx == 0 and i < 5:
                    print(f"[DEBUG L0 S{i}] Target: '{target}' | Gen: '{gen.strip()}' | Success: {is_success}")

            except Exception as e:
                print(f"Error on layer {layer_idx}: {e}")
        
        acc = success / len(data)
        results[layer_idx] = acc
        print(f"Layer {layer_idx}: {acc:.2%}")
        
        handle.remove()

    if not os.path.exists('output'): os.makedirs('output')
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f)
    
    # Diagnosis
    sorted_layers = sorted(results.items(), key=lambda x: x[1])
    print(f"\nTOP 3 BRITTLE LAYERS: {sorted_layers[:3]}")

if __name__ == "__main__":
    main()
