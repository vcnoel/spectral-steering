import torch
import importlib.util
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import get_spectral_gradient
spec = importlib.util.spec_from_file_location("calc_gradient", os.path.join(os.path.dirname(__file__), "02_calc_gradient.py"))
calc_gradient = importlib.util.module_from_spec(spec)
sys.modules["calc_gradient"] = calc_gradient
spec.loader.exec_module(calc_gradient)
get_spectral_gradient = calc_gradient.get_spectral_gradient

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
TARGET_LAYER = 2 # User request: "steer at layer 2, we proved in another article it led to success"
STEERING_STRENGTH = 1.5

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager"
    )

    prompt = "The server was accessed by the admin. The logs were checked by the auditor. The report was written by the manager. The decision"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    print(f"\n--- Regime A: Passive Voice Verification ---")
    print(f"Prompt: {prompt}")

    # Baseline Run
    print("\n[Baseline Generation]", flush=True)
    outputs = model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id, do_sample=False)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True), flush=True)

    # Steered Run with Grid Search
    strengths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 1.5]
    
    for strength in strengths:
        print(f"\n[Steered Generation (Layer {TARGET_LAYER}, Strength {strength})]", flush=True)
        
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
            
            new_hidden = hidden_states + (strength * steering_vec)
            
            if isinstance(output, tuple):
                 return (new_hidden,) + output[1:]
            else:
                 return new_hidden

        handle = model.model.layers[TARGET_LAYER].register_forward_hook(spectral_steering_hook)
        
        # Using do_sample=False (Greedy) to be deterministic
        outputs_steered = model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id, do_sample=False)
        print(tokenizer.decode(outputs_steered[0], skip_special_tokens=True), flush=True)
        
        handle.remove()

if __name__ == "__main__":
    main()
