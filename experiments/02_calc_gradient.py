import torch
from spectral_trust import GSPConfig, GraphConstructor

# Initialize engines strictly once or lazily to avoid overhead if possible, 
# but for safety in this script we init locally or globally.
_CONFIG = GSPConfig(normalization="none", symmetrization="symmetric")
_GRAPH_ENGINE = GraphConstructor(_CONFIG)

def get_spectral_gradient(hidden_states, current_attn_matrix):
    """
    Computes the gradient of the Fiedler value w.r.t hidden states using spectral-trust for graph construction.
    """
    # Enable grad if not already
    hidden_states = hidden_states.detach().requires_grad_(True)
    
    with torch.enable_grad():
        # 1. Construct Laplacian using Spectral Trust Library
        # Library expects [Batch, Heads, Seq, Seq] usually
        # We have current_attn_matrix as [Seq, Seq] numpy OR tensor?
        # current_attn_matrix comes from dummy_attn.numpy() in steering scripts. 
        # But we need it as a Tensor for graph engine if it supports torch. 
        # Let's convert to tensor.
        
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        # Prepare input: [1, 1, Seq, Seq]
        A_in = torch.tensor(current_attn_matrix, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        
        # Symmetrize (Returns [1, 1, Seq, Seq])
        A_sym = _GRAPH_ENGINE.symmetrize_attention(A_in)
        
        # Laplacian (Returns [1, Seq, Seq])
        # construct_laplacian expects [Batch, Seq, Seq] adjacency (squeezed heads).
        # We need to squeeze heads dim from A_sym
        A_adj = A_sym.squeeze(1) 
        
        L_batch = _GRAPH_ENGINE.construct_laplacian(A_adj)
        L = L_batch.squeeze(0) # [Seq, Seq]
        
        # 2. Compute Energy (Smoothness Proxy)
        X = hidden_states.squeeze(0) # [Seq, Dim]
        
        # Energy = Trace(X.T @ L @ X)
        # Note: Optimization - usually we want to minimize Fiedler eigenvalue specifically.
        # But Dirichlet energy minimization is the standard "Spectral Smoothing" proxy.
        energy = torch.trace(X.T @ L @ X)
        
        # 3. Compute Gradient
        # We want to MINIMIZE energy -> Move against gradient
        grad = torch.autograd.grad(energy, hidden_states)[0]
    
    return -grad # Points towards smoother signal
