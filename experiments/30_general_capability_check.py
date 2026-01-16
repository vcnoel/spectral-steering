import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os

# --- Configuration: The "Golden Settings" ---
# Adjust these based on your best Logic results
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
# Llama's Bi-Modal Formula (Refined from Phase 13-15):
STEERING_CONFIG = {
    15: -0.01,  # Mid-Layer Sharpen (The Texture Engine)
    26: 0.10    # Late-Layer Smooth (The Safety Valve)
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_dynamic_laplacian(activations, k=10):
    # Dynamic K-NN Graph Construction
    # Normalize for cosine similarity
    activations = torch.nn.functional.normalize(activations, p=2, dim=-1)
    # Cosine Similarity Matrix
    similarity = torch.bmm(activations, activations.transpose(1, 2))
    # Top-K
    top_k_values, _ = torch.topk(similarity, k=k, dim=-1)
    min_values = top_k_values[:, :, -1].unsqueeze(-1)
    # Adjacency Matrix
    adjacency = (similarity >= min_values).float()
    # Laplacian
    degree = adjacency.sum(dim=-1)
    laplacian = torch.diag_embed(degree) - adjacency
    return laplacian

class DynamicSteeringHook:
    def __init__(self, strength):
        self.strength = strength
    
    def __call__(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        
        # We need to compute the "Smoothness Force" (Gradient of Dirichlet Energy)
        # Gradient of L = D - A is 2 * L * X
        # We want to minimize Energy (Smooth) -> Move AGAINST Gradient (- grad)
        # So: New = Old - (strength * grad)
        # But wait, the user's snippet says: perturbed = hidden - (self.strength * smoothness_force)
        # If strength is positive (Smooth), we subtract the force? 
        # Force = 2 * L * X. This direction INCREASES roughness (it points towards high frequency).
        # So subtracting it DECREASES roughness (Smoothing). Correct.
        # If strength is negative (Sharpen), we add the force (Increasing roughness). Correct.
        
        with torch.enable_grad():
             # We need gradients for the graph construction if we want "true" spectral steering,
             # but the snippet uses a closed form "force" roughly equivalent to the gradient 
             # on a fixed graph?
             # No, the snippet computes L explicitly. 
             # The gradient of x^T L x with respect to x is (L + L^T)x. Since L is symmetric, it's 2Lx.
             # So smoothness_force = 2 * L * hidden is indeed the gradient of Dirichlet Energy.
             
             L = get_dynamic_laplacian(hidden)
             # Cast hidden to float for bmm with float L
             smoothness_force = 2 * torch.bmm(L, hidden.float())
             # Cast force back to half
             smoothness_force = smoothness_force.to(hidden.dtype)
             
             # NORMALIZED PROJECTION (SAFE MODE)
             # The force is the gradient of Energy (direction of Roughness).
             # We want to steer along the Smoothing Direction (-force).
             smoothing_direction = -smoothness_force
             spec_norm = torch.norm(smoothing_direction)
             
             if spec_norm > 1e-6:
                 # Project onto the smoothing direction with controlled magnitude
                 # Magnitude = 0.1 * |strength| * |hidden_norm|
                 # If strength is positive, we move along smoothing_dir.
                 # If strength is negative, we move against smoothing_dir (towards roughness).
                 
                 # Formula: dir / norm * hid_norm * 0.1 * strength
                 # Note: strength sign handles the direction flip.
                 perturbation = smoothing_direction / spec_norm * torch.norm(hidden) * 0.1 * self.strength
             else:
                 perturbation = torch.zeros_like(hidden)
                 
        perturbed = hidden + perturbation
        
        return (perturbed,) + output[1:] if isinstance(output, tuple) else (perturbed)

def measure_perplexity(model, tokenizer):
    print("Loading WikiText-2...")
    try:
        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception as e:
        print(f"Failed to load wikitext: {e}. Trying local fallback or subset.")
        return 999.0

    # Ensure we use a clean string join
    text_blob = "\n\n".join(test["text"][:50]) # First 50 docs for speed
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
            # Neg Log Likelihood
            nlls.append(outputs.loss * trg_len)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()

def run_safety_check():
    print(f"Loading {MODEL_NAME} for Do No Harm Check...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    
    print("--- Measuring Baseline Perplexity (General English) ---")
    baseline_ppl = measure_perplexity(model, tokenizer)
    print(f"Baseline PPL: {baseline_ppl:.2f}")
    
    print(f"--- Applying Golden Steering {STEERING_CONFIG} ---")
    hooks = []
    for layer_idx, strength in STEERING_CONFIG.items():
        layer = model.model.layers[layer_idx]
        hook = DynamicSteeringHook(strength)
        handle = layer.register_forward_hook(hook)
        hooks.append(handle)
        
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
    run_safety_check()
