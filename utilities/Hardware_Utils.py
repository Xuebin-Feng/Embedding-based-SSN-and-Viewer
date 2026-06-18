import torch

def get_optimal_device():
    """
    Returns the optimal PyTorch hardware device available on the system.
    Prioritizes locally installed specialized hardware:
    (Nvidia > Intel XPU > Apple MPS > AMD/Intel via DirectML > CPU).
    """
    # 1. Nvidia (CUDA) or AMD (ROCm via CUDA interface)
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    # 2. Intel XPU (Intel ARC/Flex extensions for PyTorch)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
        
    # 3. Apple Silicon (MPS)
    if hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
        
    # 4. AMD / Intel via Microsoft DirectML (Windows standard fallback)
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device()
    except ImportError:
        pass
        
    # 5. Last resort: CPU
    return torch.device("cpu")
