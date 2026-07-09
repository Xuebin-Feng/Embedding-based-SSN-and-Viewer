"""
File: Embedding_Injection.py
===================================
Description:
Once a network has been run, it is often useful to add a few more sequences (like a newly discovered wild-type or a 
specific reference structure) into the existing network without having to recalculate embeddings for all 50,000 original sequences.
This script takes an existing embedding HDF5 database and a NEW FASTA file containing both the old and new sequences. 
It identifies the newly added sequences, dynamically boots up the required language model, computes embeddings ONLY for the 
new additions, and synthesizes a new combined database.

Input:
- An existing HDF5 file containing pre-calculated embeddings (`OLD_HDF5`).
- A new FASTA file containing all the old sequences + any new ones you want to add (`NEW_FASTA`).

Output:
- A new, complete HDF5 file containing all embeddings, properly ordered to match the new FASTA file (`OUTPUT_HDF5`).

Settings:
- OLD_HDF5: Path to your original workspace embedding database.
- NEW_FASTA: Path to the FASTA file that contains your original sequences plus your manually added additions.
- OUTPUT_HDF5: The filename to save the newly merged database as so it does not overwrite your original.

Algorithm:
1. Loads the metadata from the old database to determine which language model (ESM/ProtT5/ProtBERT) was originally used.
2. Checks the datatypes of the arrays to ensure precision matching (FP16 vs FP32).
3. Parses the new FASTA file and performs a heavy set difference calculation to strictly identify new headers.
4. Validates that ALL original sequences are still present in the new FASTA (throwing an error if the user accidentally deleted some).
5. If new sequences are detected, it initializes the specific language model into VRAM.
6. Streams the final output database sequentially: if a sequence is old, it blitz-copies the binary array from the old file. If it is new, it routes it through the GPU for inference and streams the result directly to disk.
"""
# %% Import Necessary Libraries
import os
import re
import numpy as np
import h5py
import torch
from tqdm import tqdm
import Hardware_Utils

# Import Embedding Models
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from transformers import BertTokenizer, BertModel, T5Tokenizer, T5EncoderModel

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_EMBED = None
INPUT_FASTA = None

EMBED_DIR = os.path.join("..", "Embeddings")
FASTA_DIR = os.path.join("..", "Input_Files", "Sequence_Sets")

# --- JSON Settings Override ---
import json
import ast
import os

# Automatically calculate the root directory of the SSN project for the current PC
# (Assuming utility scripts are located in the /utilities/ folder)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SETTINGS_FILE = os.path.join(PROJECT_ROOT, "Input_Files", "tools_settings.json")

if os.path.exists(SETTINGS_FILE):
    try:
        with open(SETTINGS_FILE, "r") as f:
            all_settings = json.load(f)
            
            # 1. Load GLOBAL directories and convert relative paths to absolute paths
            if "DIRECTORIES" in all_settings:
                for k, v in all_settings["DIRECTORIES"].items():
                    if k in globals() and v is not None and str(v).strip() != "":
                        # Expand relative paths dynamically based on the current PC
                        if not os.path.isabs(str(v)):
                            v = os.path.normpath(os.path.join(PROJECT_ROOT, str(v)))
                        globals()[k] = v
                        
            # 2. Load script-specific settings
            script_name = os.path.basename(__file__)
            if script_name in all_settings:
                user_settings = all_settings[script_name]
                for k, v in user_settings.items():
                    if k in globals() and v is not None and str(v).strip() != "":
                        orig = globals()[k]
                        
                        # Type casting to match the original Python variable type
                        if isinstance(orig, int) and not isinstance(orig, bool):
                            try: v = int(v)
                            except: pass
                        elif isinstance(orig, float):
                            try: v = float(v)
                            except: pass
                        elif isinstance(orig, list):
                            try: v = ast.literal_eval(v) if isinstance(v, str) else v
                            except: pass
                        elif orig is None:
                            if v == "None": v = None
                            elif str(v).replace('.', '', 1).isdigit():
                                v = float(v) if '.' in str(v) else int(v)
                                
                        # Convert any script-specific directory paths to absolute paths
                        if isinstance(v, str) and k.endswith("_DIR") and not os.path.isabs(v):
                            v = os.path.normpath(os.path.join(PROJECT_ROOT, v))
                            
                        globals()[k] = v
    except Exception as e:
        print(f"Failed to load user settings: {e}")

# --- DYNAMIC INFERENCE ---
FULL_INPUT_EMBED = os.path.join(EMBED_DIR, INPUT_EMBED) if EMBED_DIR else ""
FULL_INPUT_FASTA = os.path.join(FASTA_DIR, INPUT_FASTA) if FASTA_DIR else ""

# %% Helper Functions
def read_fasta(file_path):
    """
    Reads a FASTA file and returns two lists: headers and sequences.
    Handles multi-line sequences correctly.
    """
    headers = []
    sequences = []
    
    current_header = None
    current_sequence = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue 
            
            if line.startswith(">"):
                if current_header is not None:
                    headers.append(current_header)
                    sequences.append("".join(current_sequence))
                
                current_header = line[1:]
                current_sequence = []
            else:
                current_sequence.append(line)
        
        if current_header is not None:
            headers.append(current_header)
            sequences.append("".join(current_sequence))
            
    return headers, sequences

def load_model(model_name):
    """
    Loads the model based on the name extracted from the HDF5 file.
    """
    device = Hardware_Utils.get_optimal_device()
    
    print(f"Loading {model_name} on {device}...")

    if "esmc" in model_name:
        client = ESMC.from_pretrained(model_name).to(device)
        return client, device, "esmc"
    
    elif "prot_bert" in model_name:
        tokenizer = BertTokenizer.from_pretrained(f"Rostlab/{model_name}", do_lower_case=False)
        model = BertModel.from_pretrained(f"Rostlab/{model_name}").to(device)
        model.eval()
        return (tokenizer, model), device, "bert"
        
    elif "ProstT5" in model_name:
        tokenizer = T5Tokenizer.from_pretrained(f"Rostlab/{model_name}_fp16", do_lower_case=False)
        model = T5EncoderModel.from_pretrained(f"Rostlab/{model_name}_fp16").to(device)
        return (tokenizer, model), device, "t5"

def get_embedding(seq, model_obj, device, model_type, target_dtype):
    """
    Generates embedding for a single sequence, casting to the inferred precision.
    Applies strict sanitization matching the base generation pipeline.
    """
    seq = seq.upper()
    
    # 1. Strip everything before the first and after the last valid AA (Standard + Extended)
    match = re.search(r'[ACDEFGHIKLMNPQRSTVWYBZJXUO].*[ACDEFGHIKLMNPQRSTVWYBZJXUO]|[ACDEFGHIKLMNPQRSTVWYBZJXUO]', seq)
    seq = match.group(0) if match else ""
    
    # 2. Model-specific internal character replacements
    if model_type == "esmc":
        # ESMC: Convert anything NOT in the standard 20 (or a gap) to '-'
        seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY\-]', '-', seq)
        
    elif model_type in ["bert", "t5"]:
        # Prot_BERT / ProstT5: Convert anything NOT standard or extended (or a gap) to 'X'
        seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWYBZJXUO\-]', 'X', seq)
        
        # Force rare B/Z/U/O tokens to X 
        seq = re.sub(r'[BZUO]', 'X', seq) 

    with torch.no_grad():
        if model_type == "esmc":
            protein_tensor = model_obj.encode(ESMProtein(sequence=seq))
            logits = model_obj.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
            return logits.embeddings.squeeze(0)[1:-1].cpu().numpy().astype(target_dtype)
            
        elif model_type == "bert":
            tokenizer, model = model_obj
            spaced_seq = " ".join(list(seq))
            inputs = tokenizer(spaced_seq, return_tensors="pt").to(device)
            outputs = model(**inputs)
            return outputs.last_hidden_state[0, 1:-1].cpu().numpy().astype(target_dtype)
            
        elif model_type == "t5":
            tokenizer, model = model_obj
            spaced_seq = " ".join(list(seq))
            input_seq = "<AA2fold> " + spaced_seq
            inputs = tokenizer(input_seq, return_tensors="pt").to(device)
            outputs = model(**inputs)
            return outputs.last_hidden_state[0, 1:-1].cpu().numpy().astype(target_dtype)

# %% Main Execution
if __name__ == "__main__":
    print("--- Step 1: Loading Original Metadata ---")
    
    if not os.path.exists(FULL_INPUT_EMBED):
        raise FileNotFoundError(f"Original embedding file not found: {FULL_INPUT_EMBED}")
        
    # Read just the metadata and headers from the old HDF5 file
    with h5py.File(FULL_INPUT_EMBED, "r") as hf_in:
        raw_headers = hf_in["headers"][:]
        old_headers = [h.decode('utf-8') if isinstance(h, bytes) else h for h in raw_headers]
        model_name = hf_in.attrs.get("model_name", "Unknown")
        
        # Infer precision dynamically based on the first existing array
        target_dtype = hf_in["embeddings"][old_headers[0]].dtype
    
    print(f"Detected Model: {model_name}")
    print(f"Detected Precision: {target_dtype}")
    print(f"Original sequences loaded: {len(old_headers)}")

    print("\n--- Step 2: Cross-Referencing Sequences ---")
    new_headers, new_seqs = read_fasta(FULL_INPUT_FASTA)
    print(f"New FASTA sequences loaded: {len(new_headers)}")
    
    old_set = set(old_headers)
    new_set = set(new_headers)
    
    # Requirement 1: Error if original headers are missing from the new FASTA
    missing_from_new = old_set - new_set
    if missing_from_new:
        example_missing = list(missing_from_new)[0]
        raise ValueError(
            f"CRITICAL ERROR: {len(missing_from_new)} sequences from the original embedding "
            f"are missing from the new FASTA file. Example missing header: '{example_missing}'"
        )
        
    # Requirement 2: Identify sequences that are in the new FASTA but not in the old file
    new_sequences_dict = {}
    for h, s in zip(new_headers, new_seqs):
        if h not in old_set:
            new_sequences_dict[h] = s
            
    print(f"Found {len(new_sequences_dict)} new sequence(s) requiring embedding generation.")

    # Load model only if new sequences exist
    if len(new_sequences_dict) > 0:
        print("\n--- Step 3: Initializing Model ---")
        model_obj, device, model_type = load_model(model_name)
    else:
        model_obj, device, model_type = None, None, None
        print("No new sequences found. Skipping model load.")

    _fasta_base = INPUT_FASTA.replace(".fasta", "")
    OUTPUT_HDF5 = os.path.join(EMBED_DIR, f"{_fasta_base}_[{model_name}]_embeddings.h5")
    
    print("\n--- Step 4: Stream Processing & Saving ---")
    os.makedirs(os.path.dirname(OUTPUT_HDF5), exist_ok=True)
    
    print(f"Reordering and streaming {len(new_headers)} synchronized embeddings...")
    
    new_generated_count = 0
    copied_count = 0
    
    with h5py.File(FULL_INPUT_EMBED, "r") as hf_in, h5py.File(OUTPUT_HDF5, "w") as hf_out:
        
        # Setup metadata and groups in the new file
        hf_out.attrs["model_name"] = model_name
        hf_out.attrs["num_sequences"] = len(new_headers)
        
        dt_str = h5py.string_dtype(encoding='utf-8')
        hf_out.create_dataset("headers", data=np.array(new_headers, dtype=object), dtype=dt_str)
        
        emb_group_out = hf_out.create_group("embeddings")
        emb_group_in = hf_in["embeddings"]
        
        # Stream the sequences in the exact order of the new FASTA
        for h, seq in tqdm(zip(new_headers, new_seqs), total=len(new_headers), desc="Writing"):
            
            if h in old_set:
                # Pluck the array directly from the old HDF5 file
                emb_data = emb_group_in[h][:]
                copied_count += 1
            else:
                # Generate it on the fly
                emb_data = get_embedding(seq, model_obj, device, model_type, target_dtype)
                new_generated_count += 1
                
            # Immediately save to the new file
            emb_group_out.create_dataset(h, data=emb_data)
        
    print(f"\n✅ Done! Injected {new_generated_count} new embeddings and copied {copied_count} existing ones.")
    print(f"Total sequences in new database: {len(new_headers)}")
    print(f"Saved directly to: {OUTPUT_HDF5}")