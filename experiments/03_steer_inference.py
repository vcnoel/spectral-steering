import torch
import json
import importlib.util
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Dynamically import the module since it starts with a number
spec = importlib.util.spec_from_file_location("calc_gradient", os.path.join(os.path.dirname(__file__), "02_calc_gradient.py"))
calc_gradient = importlib.util.module_from_spec(spec)
sys.modules["calc_gradient"] = calc_gradient
spec.loader.exec_module(calc_gradient)
get_spectral_gradient = calc_gradient.get_spectral_gradient

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
TARGET_LAYER = 12 # Set this based on findings from 01_find_fractures.py
STEERING_STRENGTH = 1.5


# Local duplicate of spectral functions to avoid import errors
def symmetrize_attention(attn_matrix):
    return 0.5 * (attn_matrix + attn_matrix.transpose())

def compute_fiedler_value(adj_matrix):
    # D = diag(sum(W))
    D = np.diag(np.sum(adj_matrix, axis=1))
    # L = D - W
    L = D - adj_matrix
    eigenvals = eigh(L, eigvals_only=True)
    if len(eigenvals) >= 2:
        return eigenvals[1]
    return 0.0

RESULTS_FILE = "output/steering_results.json"

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager"
    )
    
    # Grid Search
    strengths = [0.5, 1.0, 1.5, 2.0, 3.0]
    results = []
    
    prompt = "The server was accessed by the admin. The logs were checked by the auditor. The report"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    for alpha in strengths:
        print(f"\n--- Testing Strength {alpha} ---")
        
        # Monitor Container
        monitor_data = {"layer_13_fiedler": []}

        # 1. Steering Hook (Layer 12)
        def steer_hook(module, args, output):
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
            
            # Apply Splint
            new_hidden = hidden_states + (alpha * steering_vec)
            
            if isinstance(output, tuple):
                 return (new_hidden,) + output[1:]
            else:
                 return new_hidden

        # 2. Monitoring Hook (Layer 13)
        def monitor_hook(module, args, output):
            if isinstance(output, tuple):
                 # Attention weights are at index 1 if output_attentions=True
                 # But we might not have them if not requested.
                 # User suggested: "Just grab hidden states and approx Fiedler"
                 # OR "capture Attention Matrix of Layer 13 directly if hooking SelfAttn"
                 # Since we hook the Layer (block), output[1] contains attentions only if output_attentions=True
                 pass
            
            # The prompt says: "Simplified: Just grab hidden states and approx Fiedler"
            # But Fiedler calculation needs an adjacency matrix.
            # If we don't have attentions, we can't compute TRUE Fiedler.
            # However, in the prompt code block:
            # "Recalc Fiedler on this 'healed' signal ... compute_fiedler ..."
            # This implies we somehow get an adjacency. 
            # Given we are hooking the *LAYER*, we can't easily get the attention matrix inside the hook 
            # unless we hook the *Attention* module specifically or set output_attentions=True
            
            # Let's rely on the user's Simplified instruction: 
            # "Just grab hidden states" implies constructing a similarity graph from hidden states?
            # Or perhaps we should just assume we can get attentions?
            # Wait, the prompt code says: `monitor_data["layer_13_fiedler"].append(fiedler_val)`
            # AND `output_attentions=True` produces tuple (hidden, attentions) ?? 
            # Actually standard HF output for layer is (hidden_states, present_key_value, self_attentions)
            
            # To be safe and fast, let's use the 'dummy' adjacency approach or just skip exact Fiedler 
            # if we can't get it cheaply.
            # BUT, the goal is tuning based on Fiedler.
            # Let's hook the ATTENTION mechanism of Layer 13 instead?
            # User code: `model.model.layers[13].register_forward_hook(monitor_hook)`
            # This hooks the BLOCK.
            
            # Let's try to grab attentions if available. 
            # Warning: model.generate might not pass output_attentions=True down to layers easily 
            # without configuration.
            
            # Strategy: We will infer 'health' from the text output quality primarily?
            # No, user explicitly said: "Metric: You can't just look at text. Log the Fiedler Value..."
            
            # We will use a simplified proxy: Construct scalar product graph from hidden states
            # A_ij = <h_i, h_j>
            # This is "functional connectivity".
            
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
                
            # Compute proxy functional connectivity from hidden states
            # shape [Batch, Seq, Dim] or [Seq, Dim]
            h = hidden
            if len(h.shape) == 3: h = h[0] # Take first in batch
            
            # Normalize
            h = h / (h.norm(dim=1, keepdim=True) + 1e-8)
            # Cosine similarity
            adj = torch.mm(h, h.t()).detach().cpu().numpy()
            # Remove negative values for Laplacian ?? Fiedler needs positive weights usually?
            # Or just abs()
            adj = np.abs(adj)
            
            # Zero diagonal
            np.fill_diagonal(adj, 0)
            
            val = compute_fiedler_value(adj)
            monitor_data["layer_13_fiedler"].append(float(val))
            
            return output

        # Register Hooks
        h1 = model.model.layers[12].register_forward_hook(steer_hook)
        h2 = model.model.layers[13].register_forward_hook(monitor_hook)

        # Generate (output_attentions=False to save memory/speed, using hidden proxy for monitor)
        output_ids = model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(output_ids[0])
        
        # Calculate Metric
        if monitor_data["layer_13_fiedler"]:
            avg_health = sum(monitor_data["layer_13_fiedler"]) / len(monitor_data["layer_13_fiedler"])
        else:
            avg_health = 0.0
            
        print(f"  -> Avg Layer 13 Fiedler (Proxy): {avg_health:.4f}")
        print(f"  -> Output: {text[:50]}...")
        
        results.append({
            "strength": alpha,
            "text": text,
            "post_steering_health": avg_health
        })
        
        # Cleanup
        h1.remove()
        h2.remove()

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")
