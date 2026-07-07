"""
File: Embedding_Extraction.py
===================================
Description:
This script acts as a filter to extract a specific subset of sequence embeddings from a massive master HDF5 embedding file.
Instead of recalculating computationally expensive embeddings on a smaller subset of proteins, this script simply slices 
the existing pre-computed arrays out of the database and saves them to a new, smaller HDF5 file.

Input:
- A large source HDF5 file containing pre-computed embeddings for an entire dataset (`SOURCE_H5`).
- A target text or FASTA file containing the specific subset of sequence headers you want to extract (`TARGET_SEQUENCE_FILE`).

Output:
- A new, compact HDF5 file containing only the embeddings for the requested sequence subset (`OUTPUT_H5`).

Settings:
- SOURCE_H5: The absolute or relative path to the large embedding database.
- TARGET_SEQUENCE_FILE: The file containing the IDs or headers defining the subset to retain.
- OUTPUT_H5: The path where the new, smaller database will be written.

Algorithm:
1. Validates the existence of the source HDF5 database.
2. Parses the target text/FASTA file to generate a list of requested headers.
3. Opens the source database (Read-Only) and target database (Write) simultaneously.
4. Iterates through the requested target headers, applying filesystem sanitization rules to match the HDF5 internal structure.
5. If the header exists in the source, it copies the variable-length numpy array into the new target dataset.
6. Regenerates the master metadata arrays (`headers`, `model_name`, `num_sequences`) for the new file.
7. Logs successful transfers and explicitly prints any requested headers that were missing from the master file.
"""
# %% Imports
import h5py
import numpy as np
import os
import sys
from tqdm import tqdm

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
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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

# ==========================================
# FUNCTIONS
# ==========================================
def load_target_headers(filepath):
    """ Reads headers from a FASTA or TXT file. """
    headers = []
    ext = os.path.splitext(filepath)[1].lower()
    
    try:
        with open(filepath, "r") as f:
            if ext in [".fasta", ".fa", ".fna"]:
                print(f"Reading FASTA format: {filepath}")
                for line in f:
                    if line.startswith(">"):
                        clean_header = line.strip()[1:]
                        headers.append(clean_header)
            else:
                print(f"Reading List format: {filepath}")
                for line in f:
                    if line.strip():
                        headers.append(line.strip())
    except FileNotFoundError:
        print(f"❌ Error: Target file {filepath} not found.")
        sys.exit(1)
        
    return headers

def extract_subset():
    print(f"--- HDF5 Embedding Extractor ---")
    
    # 1. Verify Source
    if not os.path.exists(FULL_INPUT_EMBED):
        print(f"❌ Error: Source HDF5 not found at {FULL_INPUT_EMBED}")
        return

    # 2. Load Target List
    target_headers = load_target_headers(FULL_INPUT_FASTA)
    print(f"  > Target list contains {len(target_headers)} sequences.")

    found_headers = []
    missing_headers = []

    # Infer outputs
    with h5py.File(FULL_INPUT_EMBED, "r") as hf_in:
        model_name = hf_in.attrs.get("model_name", "Unknown")
        
    _fasta_base = INPUT_FASTA.replace(".fasta", "")
    OUTPUT_H5 = os.path.join(EMBED_DIR, f"{_fasta_base}_[{model_name}]_embeddings.h5")

    # 3. Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_H5), exist_ok=True)

    # 4. Open both HDF5 files (Source as Read-Only, Output as Write)
    print("Extracting matches...")
    with h5py.File(FULL_INPUT_EMBED, "r") as hf_in, h5py.File(OUTPUT_H5, "w") as hf_out:
        
        # Copy metadata
        hf_out.attrs["model_name"] = model_name
        
        emb_group_in = hf_in["embeddings"]
        emb_group_out = hf_out.create_group("embeddings")

        # Extract embeddings one by one
        for th in tqdm(target_headers, desc="Processing"):
            # Apply the exact same sanitization used during the HDF5 creation
            safe_th = th.replace("/", "_").replace("\\", "_")
            
            if safe_th in emb_group_in:
                # Read specific array into RAM, write to new file, then discard from RAM
                emb_data = emb_group_in[safe_th][:]
                emb_group_out.create_dataset(safe_th, data=emb_data)
                
                # Store the original unsanitized header for the master list
                found_headers.append(th)
            else:
                missing_headers.append(th)

        # Recreate the master headers list dataset in the new file
        dt_str = h5py.string_dtype(encoding='utf-8')
        hf_out.create_dataset("headers", data=np.array(found_headers, dtype=object), dtype=dt_str)
        hf_out.attrs["num_sequences"] = len(found_headers)

    # 5. Summary
    print(f"\n✅ Extraction Complete!")
    print(f"  > Saved to: {OUTPUT_H5}")
    print(f"  > Extracted: {len(found_headers)}")
    print(f"  > Missing:   {len(missing_headers)}")
    
    if missing_headers:
        print("\n⚠️  The following headers were not found in the source:")
        # Only print first 10 to avoid flooding the console if many are missing
        for missing in missing_headers[:10]:
            print(f"    - {missing}")
        if len(missing_headers) > 10:
            print(f"    ... and {len(missing_headers) - 10} more.")

if __name__ == "__main__":
    extract_subset()