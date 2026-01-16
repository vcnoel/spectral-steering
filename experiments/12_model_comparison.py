import torch
import json
import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from spectral_trust import GSPConfig, GraphConstructor, SpectralAnalyzer
from tqdm import tqdm

# CONFIG
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct" # The User requested "phi3 mini"
OUTPUT_FILE = "output/phi3_comparison_profile.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT = "The server was accessed by the admin. The logs were checked by the auditor. The report was written by the manager. The decision"

def main():
    if not os.path.exists('output'): os.makedirs('output')
    
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16, 
        device_map="auto", 
        attn_implementation="eager",
        trust_remote_code=True
    )

    # Initialize Spectral Trust
    config = GSPConfig(
        normalization="none", 
        symmetrization="symmetric",
        save_intermediate=False 
    )
    graph_engine = GraphConstructor(config)
    spectral_engine = SpectralAnalyzer(config)

    print(f"Profiling Passive Voice on {MODEL_NAME}...")
    
    inputs = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        
    fiedler_profile = []
    
    for i, layer_attn in enumerate(outputs.attentions):
        # Aggregation: Mean across heads [Batch, Heads, Seq, Seq] -> [Seq, Seq]
        attn = layer_attn[0].mean(dim=0)
        
        # Construct Laplacian
        attn_input = attn.unsqueeze(0).unsqueeze(0)
        adj = graph_engine.symmetrize_attention(attn_input)
        adj = adj.squeeze(1) # [1, Seq, Seq]
        L_batch = graph_engine.construct_laplacian(adj)
        L = L_batch.squeeze(0).float().cpu().numpy()
        
        # Spectral Analysis
        eigenvals, _ = spectral_engine.compute_eigendecomposition(L)
        fiedler = eigenvals[1] if len(eigenvals) >= 2 else 0.0
        fiedler_profile.append(float(fiedler))

    result = {
        "model": MODEL_NAME,
        "type": "Passive Voice",
        "fiedler_profile": fiedler_profile
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump([result], f, indent=2)
    print(f"Profile saved to {OUTPUT_FILE}")
    
    print(f"\n--- Results for {MODEL_NAME} ---")
    print(f"Fiedler Min: {min(fiedler_profile):.4f}")
    print(f"Fiedler Mean: {np.mean(fiedler_profile):.4f}")

if __name__ == "__main__":
    main()
