import re

SUPPORTED_MODELS = [
    "esm2_t6_8M",
    "esm2_t12_35M",
    "esm2_t30_150M",
    "esm2_t33_650M",
    "esm2_t36_3B",
    "esm2_t48_15B"
]

def load_model(model_name, device):
    """
    Loads the ESM-2 model and tokenizer from Hugging Face on the specified device.
    """
    from transformers import AutoTokenizer, AutoModel
    
    hf_mappings = {
        "esm2_t6_8M": "facebook/esm2_t6_8M_UR50D",
        "esm2_t12_35M": "facebook/esm2_t12_35M_UR50D",
        "esm2_t30_150M": "facebook/esm2_t30_150M_UR50D",
        "esm2_t33_650M": "facebook/esm2_t33_650M_UR50D",
        "esm2_t36_3B": "facebook/esm2_t36_3B_UR50D",
        "esm2_t48_15B": "facebook/esm2_t48_15B_UR50D"
    }
    
    hf_id = hf_mappings.get(model_name, model_name)
    print(f"Loading {model_name} ({hf_id}) on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id).to(device)
    model.eval()
    return tokenizer, model

def get_embedding(seq, model_obj, device, target_dtype):
    """
    Generates embedding for a sequence using the loaded ESM-2 model.
    """
    import torch
    tokenizer, model = model_obj

    seq = seq.upper()
    match = re.search(r'[ACDEFGHIKLMNPQRSTVWYBZJXUO].*[ACDEFGHIKLMNPQRSTVWYBZJXUO]|[ACDEFGHIKLMNPQRSTVWYBZJXUO]', seq)
    seq = match.group(0) if match else ""
    seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWYBZJXUO\-]', 'X', seq)
    seq = re.sub(r'[BZUO]', 'X', seq)

    with torch.no_grad():
        # ESM-2 takes continuous unspaced sequences
        inputs = tokenizer(seq, return_tensors="pt").to(device)
        outputs = model(**inputs)
        # ESM-2 tokenizer automatically prepends <cls> and appends <eos>, so we slice 1:-1
        return outputs.last_hidden_state[0, 1:-1].cpu().numpy().astype(target_dtype)
