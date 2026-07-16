import re

SUPPORTED_MODELS = ["esmc_300m", "esmc_600m"]

def load_model(model_name, device):
    """
    Loads the ESMC model on the specified device.
    """
    from esm.models.esmc import ESMC
    print(f"Loading {model_name} on {device}...")
    client = ESMC.from_pretrained(model_name).to(device)
    return client

def get_embedding(seq, model_obj, device, target_dtype):
    """
    Generates embedding for a sequence using the loaded ESMC model.
    """
    import torch
    from esm.sdk.api import ESMProtein, LogitsConfig

    seq = seq.upper()
    match = re.search(r'[ACDEFGHIKLMNPQRSTVWYBZJXUO].*[ACDEFGHIKLMNPQRSTVWYBZJXUO]|[ACDEFGHIKLMNPQRSTVWYBZJXUO]', seq)
    seq = match.group(0) if match else ""
    seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY\-]', '-', seq)

    with torch.no_grad():
        protein_tensor = model_obj.encode(ESMProtein(sequence=seq))
        logits = model_obj.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
        # Slice out the start/end special tokens and convert to target precision
        return logits.embeddings.squeeze(0)[1:-1].cpu().numpy().astype(target_dtype)
