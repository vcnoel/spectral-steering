import torch
import json
import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from spectral_trust import GSPConfig, GraphConstructor, SpectralAnalyzer
from tqdm import tqdm

# CONFIG
MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
OUTPUT_FILE = "output/full_spectral_profile.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPTS = [
    {
        "type": "Nested Logic (Fracture)",
        "text": "If P implies Q, and Q implies R, and R implies S. P is true. However, S is false. This creates a contradiction because"
    },
    {
        "type": "Healthy Baseline",
        "text": "The quick brown fox jumps over the lazy dog. The sun is shining today. I love to code in Python."
    }
]

def main():
    if not os.path.exists('output'): os.makedirs('output')
    
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager"
    )

    # Initialize Spectral Trust
    config = GSPConfig(
        normalization="none", 
        symmetrization="symmetric",
        save_intermediate=False 
    )
    graph_engine = GraphConstructor(config)
    spectral_engine = SpectralAnalyzer(config)

    results = []
    
    for item in tqdm(PROMPTS):
        prompt = item['text']
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
            
        # We need hidden states for signal analysis (Smoothness, Entropy, HFER)
        # Hidden states usually tuple of (batch, seq, dim)
        # outputs.hidden_states contains (embed, layer_1, ... layer_N)
        # We also need Attention for Laplacian.
        
        layer_metrics = {
            "fiedler_profile": [],
            "smoothness_profile": [],
            "entropy_profile": [],
            "hfer_profile": []
        }
        
        # Phi-3 has 32 layers. outputs.attentions has 32 items. outputs.hidden_states has 33 items (embedding + 32).
        # We'll analyze layers 0-31 corresponding to attention blocks.
        
        num_layers = len(outputs.attentions)
        
        for i in range(num_layers):
            # 1. Signals (Hidden States at LAYER INPUT or OUTPUT?)
            # Usually we analyze the hidden state *after* the layer processing or *input* to it. 
            # Let's use the hidden state *output* of layer i (index i+1 in tuple).
            hidden_state = outputs.hidden_states[i+1] # [1, Seq, Dim]
            signals = hidden_state.squeeze(0) # [Seq, Dim]
            
            # 2. Laplacian (From Attention)
            attn_matrix = outputs.attentions[i][0].mean(dim=0) # Mean head allocation [Seq, Seq]
            
            # Use Library to construct Laplacian
            # Requires [1, 1, Seq, Seq] input for symmetrize
            attn_input = attn_matrix.unsqueeze(0).unsqueeze(0)
            adj = graph_engine.symmetrize_attention(attn_input)
            adj = adj.squeeze(1) # [1, Seq, Seq]
            L_batch = graph_engine.construct_laplacian(adj)
            laplacian = L_batch.squeeze(0) # [Seq, Seq]
            
            # 3. Analyze
            diagnostics = spectral_engine.analyze_layer(signals, laplacian, i)
            
            layer_metrics["fiedler_profile"].append(float(diagnostics.fiedler_value))
            layer_metrics["smoothness_profile"].append(float(diagnostics.smoothness_index))
            layer_metrics["entropy_profile"].append(float(diagnostics.spectral_entropy))
            layer_metrics["hfer_profile"].append(float(diagnostics.hfer))

        results.append({
            "type": item['type'],
            "metrics": layer_metrics
        })

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Full profile saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
