import torch
import importlib.util
import sys
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import get_spectral_gradient
spec = importlib.util.spec_from_file_location("calc_gradient", os.path.join(os.path.dirname(__file__), "02_calc_gradient.py"))
calc_gradient = importlib.util.module_from_spec(spec)
sys.modules["calc_gradient"] = calc_gradient
spec.loader.exec_module(calc_gradient)
get_spectral_gradient = calc_gradient.get_spectral_gradient

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

# Prompts designed to test "Nested Logic" and "Depth"
PROMPTS = [
    {
        "id": "original_implication",
        "text": "If P implies Q, and Q implies R, and R implies S. P is true. However, S is false. This creates a contradiction because"
    },
    {
        "id": "depth_4_chain",
        "text": "Rule 1: If the red light is on, the door is locked. Rule 2: If the door is locked, the key is hidden. Rule 3: If the key is hidden, the guard is sleeping. Rule 4: If the guard is sleeping, the alarm is off. We observe that the Red Light is ON, but the Alarm is ON. This is impossible because"
    },
    {
        "id": "nested_categories",
        "text": "All A are B. All B are C. All C are D. Some D are E. No E are A. If we find an object that is an A, can it be an E? Explain step by step:"
    },
    {
        "id": "temporal_sequence",
        "text": "Event A happened before Event B. Event B happened before Event C. Event C happened after Event D but before Event A. This sequence is logically impossible because"
    }
]

LAYERS = [16, 18, 20, 22, 24]
STRENGTHS = [1.0, 1.5, 2.0]
OUTPUT_FILE = "output/logic_robustness_results.json"

def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager"
    )

    results = []

    for prompt_data in PROMPTS:
        pid = prompt_data["id"]
        text = prompt_data["text"]
        
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        print(f"\n=== Testing Prompt: {pid} ===")

        # Baseline
        print("[Baseline]")
        base_out = model.generate(**inputs, max_new_tokens=60, pad_token_id=tokenizer.eos_token_id, do_sample=False)
        base_text = tokenizer.decode(base_out[0], skip_special_tokens=True)
        print(base_text[len(text):].strip()[:100] + "...")
        
        results.append({
            "prompt_id": pid,
            "type": "baseline",
            "layer": None,
            "strength": None,
            "output": base_text[len(text):]
        })

        # Sweep
        for layer in LAYERS:
            for strength in STRENGTHS:
                print(f"[Layer {layer} | Strength {strength}]")
                
                def hook(module, args, output):
                    if isinstance(output, tuple): h = output[0]
                    else: h = output
                    
                    if len(h.shape) == 3: seq_len = h.shape[1]
                    else: seq_len = h.shape[0]
                    
                    dummy_attn = torch.ones((seq_len, seq_len))/seq_len
                    grad = get_spectral_gradient(h, dummy_attn.numpy())
                    new_h = h + (strength * grad)
                    
                    if isinstance(output, tuple): return (new_h,) + output[1:]
                    else: return new_h

                handle = model.model.layers[layer].register_forward_hook(hook)
                
                try:
                    out = model.generate(**inputs, max_new_tokens=60, pad_token_id=tokenizer.eos_token_id, do_sample=False)
                    gen_text = tokenizer.decode(out[0], skip_special_tokens=True)
                    print(f" -> {gen_text[len(text):].strip()[:100]}...")
                    
                    results.append({
                        "prompt_id": pid,
                        "type": "steered",
                        "layer": layer,
                        "strength": strength,
                        "output": gen_text[len(text):]
                    })
                except Exception as e:
                    print(f"Error: {e}")
                finally:
                    handle.remove()

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
