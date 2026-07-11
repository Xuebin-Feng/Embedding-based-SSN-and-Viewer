import re

SUPPORTED_MODELS = [
    "ankh_base",
    "ankh_large"
]

def load_model(model_name, device):
    """
    Loads the Ankh model (encoder-only) and tokenizer on the specified device.
    """
    from transformers import AutoTokenizer, T5EncoderModel
    
    hf_mappings = {
        "ankh_base": "ElnaggarLab/ankh-base",
        "ankh_large": "ElnaggarLab/ankh-large"
    }
    
    hf_id = hf_mappings.get(model_name, model_name)
    print(f"Loading {model_name} ({hf_id}) on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = T5EncoderModel.from_pretrained(hf_id).to(device)
    model.eval()
    return tokenizer, model

def get_embedding(seq, model_obj, device, target_dtype):
    """
    Generates embedding for a sequence using the loaded Ankh model.
    """
    import torch
    tokenizer, model = model_obj

    seq = seq.upper()
    match = re.search(r'[ACDEFGHIKLMNPQRSTVWYBZJXUO].*[ACDEFGHIKLMNPQRSTVWYBZJXUO]|[ACDEFGHIKLMNPQRSTVWYBZJXUO]', seq)
    seq = match.group(0) if match else ""
    seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWYBZJXUO\-]', 'X', seq)
    seq = re.sub(r'[BZUO]', 'X', seq)

    with torch.no_grad():
        # Ankh tokenizer takes unspaced sequences and appends only </s> at the end.
        inputs = tokenizer(seq, return_tensors="pt").to(device)
        outputs = model(**inputs)
        # Slicing :-1 drops the trailing </s> and retains the exact residue representations.
        return outputs.last_hidden_state[0, :-1].cpu().numpy().astype(target_dtype)
