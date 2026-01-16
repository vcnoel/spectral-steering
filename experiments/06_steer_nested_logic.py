import torch
import numpy as np
import importlib.util
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.linalg import eigh

# Import get_spectral_gradient
spec = importlib.util.spec_from_file_location("calc_gradient", os.path.join(os.path.dirname(__file__), "02_calc_gradient.py"))
calc_gradient = importlib.util.module_from_spec(spec)
sys.modules["calc_gradient"] = calc_gradient
spec.loader.exec_module(calc_gradient)
get_spectral_gradient = calc_gradient.get_spectral_gradient

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
TARGET_LAYER = 2 # Steering at Layer 2 for Logic
STEERING_STRENGTH = 1.5 

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager"
    )

    prompt = "If P implies Q, and Q implies R, and R implies S. P is true. However, S is false. This creates a contradiction because"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    print(f"\n--- Nested Logic Repair Experiment ---")
    print(f"Prompt: {prompt}")

    # Baseline Run
    print("\n[Baseline Generation]", flush=True)
    outputs = model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id, do_sample=False)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True), flush=True)

    # Layer Sweep for "Synthesis Phase" Hypothesis
    layers_to_test = [10, 12, 14, 16, 18, 20] 
    
    for layer in layers_to_test:
        print(f"\n[Steered Generation (Layer {layer}, Strength {STEERING_STRENGTH})]", flush=True)
        
        def spectral_steering_hook(module, args, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
                
            if len(hidden_states.shape) == 3:
                seq_len = hidden_states.shape[1]
            else:
                seq_len = hidden_states.shape[0]
                
            dummy_attn = torch.ones((seq_len, seq_len)) / seq_len
            steering_vec = get_spectral_gradient(hidden_states, dummy_attn.numpy())
            
            new_hidden = hidden_states + (STEERING_STRENGTH * steering_vec)
            
            if isinstance(output, tuple):
                 return (new_hidden,) + output[1:]
            else:
                 return new_hidden

        handle = model.model.layers[layer].register_forward_hook(spectral_steering_hook)
        
        outputs_steered = model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id, do_sample=False)
        print(tokenizer.decode(outputs_steered[0], skip_special_tokens=True), flush=True)
        
        handle.remove()

if __name__ == "__main__":
    main()
