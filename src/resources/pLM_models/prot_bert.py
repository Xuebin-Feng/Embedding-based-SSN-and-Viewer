import re

SUPPORTED_MODELS = ["prot_bert"]

def load_model(model_name, device):
    """
    Loads the ProtBERT model and tokenizer on the specified device.
    """
    from transformers import BertTokenizer, BertModel
    print(f"Loading {model_name} on {device}...")
    tokenizer = BertTokenizer.from_pretrained(f"Rostlab/{model_name}", do_lower_case=False)
    model = BertModel.from_pretrained(f"Rostlab/{model_name}").to(device)
    model.eval()
    return tokenizer, model

def get_embedding(seq, model_obj, device, target_dtype):
    """
    Generates embedding for a sequence using the loaded ProtBERT model.
    """
    import torch
    tokenizer, model = model_obj

    seq = seq.upper()
    match = re.search(r'[ACDEFGHIKLMNPQRSTVWYBZJXUO].*[ACDEFGHIKLMNPQRSTVWYBZJXUO]|[ACDEFGHIKLMNPQRSTVWYBZJXUO]', seq)
    seq = match.group(0) if match else ""
    # Replace non-standard amino acids
    seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWYBZJXUO\-]', 'X', seq)
    seq = re.sub(r'[BZUO]', 'X', seq)

    with torch.no_grad():
        spaced_seq = " ".join(list(seq))
        inputs = tokenizer(spaced_seq, return_tensors="pt").to(device)
        outputs = model(**inputs)
        # Slice out start/end special tokens ([CLS]/[SEP]) and convert to target precision
        return outputs.last_hidden_state[0, 1:-1].cpu().numpy().astype(target_dtype)
