import re

SUPPORTED_MODELS = ["ProstT5"]

def load_model(model_name, device):
    """
    Loads the ProstT5 model and tokenizer on the specified device.
    """
    from transformers import T5Tokenizer, T5EncoderModel
    print(f"Loading {model_name} on {device}...")
    tokenizer = T5Tokenizer.from_pretrained(f"Rostlab/{model_name}_fp16", do_lower_case=False)
    model = T5EncoderModel.from_pretrained(f"Rostlab/{model_name}_fp16").to(device)
    return tokenizer, model

def get_embedding(seq, model_obj, device, target_dtype):
    """
    Generates embedding for a sequence using the loaded ProstT5 model.
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
        input_seq = "<AA2fold> " + spaced_seq
        inputs = tokenizer(input_seq, return_tensors="pt").to(device)
        outputs = model(**inputs)
        # Slice out start/end special tokens and convert to target precision
        return outputs.last_hidden_state[0, 1:-1].cpu().numpy().astype(target_dtype)
