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
TARGET_LAYER = 2 # User identified fracture at Layer 2 via visual analysis
STEERING_STRENGTH = 1.5 # Using robust default from tuning

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager"
    )

    prompt = "All wugs are cute. This animal is not cute. Therefore, this animal is definitely not a"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    print(f"\n--- Logic Repair Experiment ---")
    print(f"Prompt: {prompt}")

    # Baseline Run
    print("\n[Baseline Generation]")
    outputs = model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(outputs[0]))

    # Steered Run
    print(f"\n[Steered Generation (Strength {STEERING_STRENGTH})]")
    
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

    handle = model.model.layers[TARGET_LAYER].register_forward_hook(spectral_steering_hook)
    
    outputs_steered = model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(outputs_steered[0]))
    
    handle.remove()

if __name__ == "__main__":
    main()
