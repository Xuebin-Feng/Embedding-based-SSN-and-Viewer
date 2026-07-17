"""
File: Generate_Embeddings.py
===================================
Description:
This script acts as the foundation for the structural similarity pipeline. It ingests a standard FASTA file 
containing raw amino acid sequences and utilizes large language models (pLM) to convert each sequence into 
a mathematically dense, high-dimensional floating-point representation (embedding).

Input:
- A text-based FASTA file containing raw protein sequence strings (`INPUT_FASTA`).

Output:
- A comprehensive, serialized HDF5 database containing the structural embedding arrays for every sequence, metadata, and 
precisely matched order arrays (`OUTPUT_HDF5`).

Settings:
- SEQUENCE_SET: Defines the input FASTA file to target.
- MODEL_NAME: The protein language model identifier to download from HuggingFace and load into VRAM. Supported models include 
  the Evolutionary Scale Modeling families (`esmc_300m`, `esmc_600m`), and the Rostlab families (`prot_bert`, `ProstT5`).
- SAVING_MODE: Determines data precision. `float16` halves HDF5 file size and RAM requirements by slightly reducing gradient precision, 
  which is recommended for massive datasets. `float32` uses standard uncompressed precision.

Algorithm:
1. Sequentially parses the target FASTA string blocks into RAM.
2. Identifies PyTorch hardware acceleration (CUDA/XPU/CPU) and allocates the massive neural networks accordingly.
3. Initializes a new HDF5 file stream in append mode ("a"), checking for existing embeddings to allow seamless resuming.
4. Iterates linearly over sequences. Each sequence is cleaned (gaps removed) and passed into the loaded neural network.
5. The model strips start/stop tokens internally and isolates the output matrices characterizing every residue in the sequence.
6. The resultant PyTorch tensor is demoted to a Numpy array, cast to the selected precision (`float16/float32`), and streamed 
   directly to disk under a sanitized header name key to prevent RAM overflow.
"""
# %% Import Necessary Libraries
import os
from tqdm import tqdm
import numpy as np
import torch
import h5py
import re
import Hardware_Utils

# Script configuration
INPUT_FASTA = None
MODEL_NAME = None
SAVING_MODE = "float16" 
                  
FASTA_DIR = os.path.join("..", "Input_Files", "Sequence_Sets")
EMBED_DIR = os.path.join("..", "Embeddings")

# --- JSON Settings Override ---
import json
import ast

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
                        if isinstance(v, str) and k.endswith("_DIR") and not os.path.isabs(v):
                            v = os.path.normpath(os.path.join(PROJECT_ROOT, v))
                        globals()[k] = v
    except Exception as e:
        print(f"Failed to load user settings: {e}")

# --- DYNAMIC PATH INFERENCE ---
FULL_INPUT_FASTA = os.path.join(FASTA_DIR, INPUT_FASTA) if FASTA_DIR and INPUT_FASTA else ""

# Derive the base name for saving
SEQUENCE_SET = INPUT_FASTA.replace(".fasta", "") if INPUT_FASTA else "Unknown_Set"
OUTPUT_HDF5 = os.path.join(EMBED_DIR, f"{SEQUENCE_SET}_[{MODEL_NAME}]_embeddings.h5") if EMBED_DIR else ""

# Embedding model imports are deferred to dynamic plugin scripts under src/resources/pLM_models/

# Helper function
def read_fasta(file_path):
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

# %% =======================================
# OPTIMIZED EMBEDDING (GPU/CPU)
# ==========================================

def find_model_plugin(model_name):
    """
    Dynamically locates and loads the plugin script supporting the selected model_name.
    Uses AST to inspect supported models statically to avoid running code of non-matching plugins.
    """
    import ast
    import glob
    import importlib.util

    current_dir = os.path.dirname(os.path.abspath(__file__))
    plugin_dir = os.path.abspath(os.path.join(current_dir, "..", "resources", "pLM_models"))

    if not os.path.exists(plugin_dir):
        raise FileNotFoundError(f"Plugin directory not found: {plugin_dir}")

    for filepath in glob.glob(os.path.join(plugin_dir, "*.py")):
        if os.path.basename(filepath) == "__init__.py":
            continue
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                node = ast.parse(f.read(), filename=filepath)
            
            supported_models = []
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id == "SUPPORTED_MODELS":
                            supported_models = ast.literal_eval(item.value)
                            break
            
            if model_name in supported_models:
                module_name = os.path.splitext(os.path.basename(filepath))[0]
                spec = importlib.util.spec_from_file_location(module_name, filepath)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
        except Exception as e:
            print(f"Warning: Failed to parse/load plugin {filepath}: {e}")
    return None

# %% =======================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print(f"--- Step 1: Embedding Generation ---")
    
    # 1. Configuration Check
    if SAVING_MODE == "float16":
        target_dtype = np.float16
        print("--> Mode: Saving as float16 (Compact)")
    elif SAVING_MODE == "float32":
        target_dtype = np.float32
        print("--> Mode: Saving as float32 (High Precision)")
    else:
        raise ValueError("SAVING_MODE must be 'float16' or 'float32'")

    # 2. Read Data
    headers, seqs = read_fasta(FULL_INPUT_FASTA)
    print(f"Loaded {len(headers)} sequences from FASTA.")
    
    os.makedirs(os.path.dirname(OUTPUT_HDF5), exist_ok=True)
    print(f"Opening {OUTPUT_HDF5} for evaluation...")
    
    # 3. Check Completeness & Resume Logic using "a" (Append) Mode
    with h5py.File(OUTPUT_HDF5, "a") as hf:
        
        # Ensure the group exists
        if "embeddings" not in hf:
            emb_group = hf.create_group("embeddings")
        else:
            emb_group = hf["embeddings"]
            
        existing_keys = set(emb_group.keys())
        
        pending_headers = []
        pending_seqs = []
        
        # Build a list of sequences that actually need to be generated
        for h, s in zip(headers, seqs):
            safe_h = h.replace("/", "_").replace("\\", "_")
            if safe_h not in existing_keys:
                pending_headers.append(h)
                pending_seqs.append(s)
                
        if len(pending_headers) == 0:
            print(f"✅ HDF5 database already complete ({len(headers)} embeddings). Skipping generation.")
        else:
            if len(pending_headers) < len(headers):
                print(f"🔄 Resuming from interruption: {len(existing_keys)} found, {len(pending_headers)} remaining.")
            else:
                print(f"🚀 Starting fresh embedding generation for {len(pending_headers)} sequences.")
                
            # 4. Find and Load Model Plugin (Only if we actually have work to do!)
            plugin = find_model_plugin(MODEL_NAME)
            if plugin is None:
                raise ValueError(f"Model '{MODEL_NAME}' is not supported by any available plugin in 'pLM_models'.")
            
            device = Hardware_Utils.get_optimal_device()
            model_obj = plugin.load_model(MODEL_NAME, device)
            
            # 5. Generate & Save Loop
            for header, seq in tqdm(zip(pending_headers, pending_seqs), total=len(pending_headers), desc="Embedding"):
                emb = plugin.get_embedding(seq, model_obj, device, target_dtype)
                safe_header = header.replace("/", "_").replace("\\", "_")
                emb_group.create_dataset(safe_header, data=emb)
        
        # 6. Finalize Metadata (Overwriting to ensure the master list perfectly matches the current FASTA file)
        hf.attrs["model_name"] = MODEL_NAME
        hf.attrs["num_sequences"] = len(headers)
        
        if "headers" in hf:
            del hf["headers"] # Delete the old array before replacing it
            
        dt_str = h5py.string_dtype(encoding='utf-8')
        hf.create_dataset("headers", data=np.array(headers, dtype=object), dtype=dt_str)

    if len(pending_headers) > 0:
        print(f"\n✅ Done! All embeddings generated and saved to {OUTPUT_HDF5}")