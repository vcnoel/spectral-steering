import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
# CONFIG
MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
OUTPUT_FILE = "output/fracture_log.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Library Imports
from spectral_trust import GSPConfig, GraphConstructor, SpectralAnalyzer
from tqdm import tqdm

def main():
    if not os.path.exists('output'): os.makedirs('output')
    
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager"
    )

    # Initialize Spectral Trust
    config = GSPConfig(eigen_solver="dense", normalization="none", symmetrization="symmetric")
    graph_engine = GraphConstructor(config)
    spectral_engine = SpectralAnalyzer(config)

    with open('data/stress_test.json', 'r') as f:
        data = json.load(f)

    results = []
    
    for item in tqdm(data):
        prompt = item['prompt']
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            
        layer_fiedlers = []
        for i, layer_attn in enumerate(outputs.attentions):
            # Aggregation: Mean across heads [Batch, Heads, Seq, Seq] -> [Seq, Seq]
            # Need to select batch 0 first
            attn = layer_attn[0].mean(dim=0) # [Seq, Seq]
            
            # Add Batch/Head dims for library compatibility if needed, or pass raw if expected
            # Library expects: [batch, heads, seq, seq] OR handles lower dims?
            # Let's assume we pass [Seq, Seq] and library handles it, OR we match library signature.
            # Library signature seen in Step 182 learnings: `symmetrize_attention(self, attention: torch.Tensor)`
            # and returns tensor. 
            
            # Let's adapt to library input expected shape.
            # Usually libraries expect batched input.
            # Let's pass [1, 1, Seq, Seq] just to be safe or check source?
            # Step 182 snippet: "Args: attention: [batch, heads, seq_len, seq_len]"
            
            attn_input = attn.unsqueeze(0).unsqueeze(0) # [1, 1, Seq, Seq]
            
            # Symmetrize
            adj = graph_engine.symmetrize_attention(attn_input) # Returns [1, 1, Seq, Seq]
            adj = adj.squeeze() # [Seq, Seq]
            
            # Laplacian
            # construct_laplacian expects [batch, seq, seq]? 
            # Step 182: "Args: adjacency: [batch, seq_len, seq_len]"
            l_input = adj.unsqueeze(0) # [1, Seq, Seq]
            L = graph_engine.construct_laplacian(l_input) # Returns [1, Seq, Seq]
            L = L.squeeze(0).float().cpu().numpy()
            
            # Spectral Analysis
            eigenvals, _ = spectral_engine.compute_eigendecomposition(L)
            
            # Fiedler is index 1
            if len(eigenvals) >= 2:
                fiedler = eigenvals[1]
            else:
                fiedler = 0.0
            
            layer_fiedlers.append(float(fiedler))
            
            # Simple fracture detection heuristic
            if fiedler < 0.05 and i > 5:
                print(f"Fracture detected in '{item['type']}' at Layer {i}: {fiedler:.4f}")

        results.append({
            "type": item['type'],
            "fiedler_profile": layer_fiedlers
        })

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Diagnosis saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
