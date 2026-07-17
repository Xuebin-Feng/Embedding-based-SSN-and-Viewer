"""
File: Embedding_Cropping.py
===================================
Description:
This script produces contextually accurate embeddings for cropped/partial protein sequences by slicing
them out of an already-computed full-sequence embedding database, instead of embedding the cropped
sequence in isolation. Because transformer-based protein language models compute every residue's
representation using full self-attention context, embedding a short fragment directly yields a different
(context-impoverished) vector than the same residues would get inside their native full-length sequence.
This script avoids that discrepancy: it never re-runs the language model, it only reads an existing
full-sequence embedding file and slices out the requested residue range for each cropped sequence.

Input:
- A pre-computed full-sequence embedding database (`INPUT_EMBED`), generated separately via
  `Generate_Embeddings.py`.
- The full-length FASTA file (`FULL_FASTA`) that `INPUT_EMBED` was generated from.
- A FASTA file of cropped/partial sequences (`CROPPED_FASTA`), one entry per header, where each sequence
  is a contiguous substring of the corresponding full sequence in `FULL_FASTA`.

Output:
- A new HDF5 database (`OUTPUT_HDF5`) containing the cropped embedding slice for each resolved header, in
  the same format `Generate_Embeddings.py` would produce if run directly on `CROPPED_FASTA`.

Settings:
- INPUT_EMBED: The pre-existing full-sequence embeddings (.h5) to slice from.
- FULL_FASTA: The full-length sequence set matching `INPUT_EMBED`.
- CROPPED_FASTA: The cropped/partial sequence set to produce contextual embeddings for.

Requirements:
- `FULL_FASTA` and `INPUT_EMBED` do not need to match `CROPPED_FASTA` (or each other) in size — extra
  entries in either are simply ignored. The only requirement is that every header present in
  `CROPPED_FASTA` also exists in both `FULL_FASTA` and `INPUT_EMBED`.
- Sequences are matched by exact substring search (`full_seq.find(crop_seq)`), so the cropped sequence's
  characters must appear identically within the full sequence. If a match occurs more than once, the
  first occurrence is used and a warning is logged.

Algorithm:
1. Parses `FULL_FASTA` and `CROPPED_FASTA` into header->sequence dictionaries.
2. Opens `INPUT_EMBED` read-only and reads its `model_name` attribute for output metadata/naming.
3. For every header in `CROPPED_FASTA`: locates the matching full sequence and its full-length embedding
   array, validates that the embedding's row count matches the full sequence's length (catches a
   mismatched/stale `INPUT_EMBED`), finds the crop's offset via substring search, and slices the
   corresponding rows out of the full embedding.
4. Writes all resolved slices into a new HDF5 file using the same `embeddings`/`headers`/`attrs`
   structure as `Generate_Embeddings.py`.
5. Prints a summary of how many headers were successfully cropped versus skipped, broken down by reason
   (missing from full FASTA, missing from embedding file, length mismatch, substring not found, or
   ambiguous match resolved via first occurrence).
"""
# %% Import Necessary Libraries
import os
from tqdm import tqdm
import numpy as np
import h5py

# Script configuration
INPUT_EMBED = None
FULL_FASTA = None
CROPPED_FASTA = None

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
FULL_INPUT_EMBED = os.path.join(EMBED_DIR, INPUT_EMBED) if EMBED_DIR and INPUT_EMBED else ""
FULL_FASTA_PATH = os.path.join(FASTA_DIR, FULL_FASTA) if FASTA_DIR and FULL_FASTA else ""
CROPPED_FASTA_PATH = os.path.join(FASTA_DIR, CROPPED_FASTA) if FASTA_DIR and CROPPED_FASTA else ""

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

    return dict(zip(headers, sequences))

def safe_key(header):
    return header.replace("/", "_").replace("\\", "_")

# %% =======================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print(f"--- Embedding Cropping ---")

    # 1. Read Data
    full_seqs = read_fasta(FULL_FASTA_PATH)
    cropped_seqs = read_fasta(CROPPED_FASTA_PATH)
    print(f"Loaded {len(full_seqs)} full sequences and {len(cropped_seqs)} cropped sequences.")

    if not os.path.exists(FULL_INPUT_EMBED):
        raise FileNotFoundError(f"Input embedding database not found: {FULL_INPUT_EMBED}")

    # 2. Resolve crop offsets and slice embeddings
    missing_from_fasta = []
    missing_from_embed = []
    length_mismatch = []
    substring_not_found = []
    ambiguous_resolved = []
    resolved = []  # list of (header, cropped_embedding_array)

    with h5py.File(FULL_INPUT_EMBED, "r") as hf_in:
        model_name = hf_in.attrs.get("model_name", "Unknown")
        emb_group_in = hf_in["embeddings"]

        for header, crop_seq in tqdm(cropped_seqs.items(), total=len(cropped_seqs), desc="Cropping"):
            if header not in full_seqs:
                missing_from_fasta.append(header)
                continue
            full_seq = full_seqs[header]

            safe_h = safe_key(header)
            if safe_h in emb_group_in:
                full_emb = emb_group_in[safe_h][:]
            elif header in emb_group_in:
                full_emb = emb_group_in[header][:]
            else:
                missing_from_embed.append(header)
                continue

            if full_emb.shape[0] != len(full_seq):
                length_mismatch.append(header)
                continue

            offset = full_seq.find(crop_seq)
            if offset == -1:
                substring_not_found.append(header)
                continue
            if full_seq.count(crop_seq) > 1:
                ambiguous_resolved.append(header)

            cropped_emb = full_emb[offset: offset + len(crop_seq)]
            resolved.append((header, cropped_emb))

    # 3. Write Output
    _cropped_base = CROPPED_FASTA.replace(".fasta", "") if CROPPED_FASTA else "Unknown_Set"
    OUTPUT_HDF5 = os.path.join(EMBED_DIR, f"{_cropped_base}_[{model_name}]_embeddings.h5")
    os.makedirs(os.path.dirname(OUTPUT_HDF5), exist_ok=True)

    with h5py.File(OUTPUT_HDF5, "w") as hf_out:
        hf_out.attrs["model_name"] = model_name

        emb_group_out = hf_out.create_group("embeddings")
        resolved_headers = []
        for header, cropped_emb in resolved:
            emb_group_out.create_dataset(safe_key(header), data=cropped_emb)
            resolved_headers.append(header)

        dt_str = h5py.string_dtype(encoding='utf-8')
        hf_out.create_dataset("headers", data=np.array(resolved_headers, dtype=object), dtype=dt_str)
        hf_out.attrs["num_sequences"] = len(resolved_headers)

    # 4. Summary
    print(f"\n✅ Done! Cropped embeddings saved to {OUTPUT_HDF5}")
    print(f"  > Resolved: {len(resolved_headers)} / {len(cropped_seqs)}")

    if ambiguous_resolved:
        print(f"\n⚠️  {len(ambiguous_resolved)} header(s) had an ambiguous (repeated) crop match — used first occurrence:")
        for h in ambiguous_resolved[:10]:
            print(f"    - {h}")
        if len(ambiguous_resolved) > 10:
            print(f"    ... and {len(ambiguous_resolved) - 10} more.")

    def _report(label, items):
        if not items:
            return
        print(f"\n⚠️  {len(items)} header(s) skipped ({label}):")
        for h in items[:10]:
            print(f"    - {h}")
        if len(items) > 10:
            print(f"    ... and {len(items) - 10} more.")

    _report("missing from full FASTA", missing_from_fasta)
    _report("missing from embedding file", missing_from_embed)
    _report("length mismatch between embedding and full sequence", length_mismatch)
    _report("crop substring not found in full sequence", substring_not_found)
