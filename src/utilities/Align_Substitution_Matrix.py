"""
File: Align_Substitution_Matrix.py
===================================
Description:
This script computes an all-vs-all sequence similarity network utilizing traditional NCBI BLASTP logic.
It formats the input FASTA file into a local BLAST database and aligns it against itself to compute 
substitution matrix similarity metrics.

Input:
- A FASTA file containing all sequences to be aligned (`{SEQUENCE_SET}.fasta`).

Output:
- An HDF5 file containing the list of pairwise edges, their source/target indices, and their negative Log10(E-Value) score (`{SEQUENCE_SET}_BLAST_EValue.h5`).

Settings:
- SEQUENCE_SET: The base name of the sequence group.
- MATRIX: The amino acid substitution matrix to use for scoring (e.g. "BLOSUM62", "PAM30").
- NUM_THREADS: Number of parallel multiprocessing workers/cores to spawn for BLAST.
- E_VALUE_CUTOFF: The maximum statistical E-value threshold to save an alignment. Alignments worse than this are ignored.
- MAX_TARGET_SEQS: Limits the total number of hits returned per query sequence. Note that BLAST may cut off valid alignments if this is too low.
- BATCH_SIZE: Number of parsing elements kept in RAM before committing to a temp pickle file to avoid Out of Memory errors.

Algorithm:
1. Validates the FASTA inputs and initializes the workspace by clearing out any old run fragments if settings have changed.
2. Uses sequence splitting algorithms to partition the FASTA query into parallel chunks.
3. Prepares the target dataset by spawning an `NCBI makeblastdb` background task.
4. Concurrently maps the `blastp` binary across all query chunks. Output format is enforced as a tabular 12 column format strings.
5. Recursively parses the output chunk text files to match sequence IDs back to an integer map.
6. Converts resulting statistical E-Values into mathematical -Log10(E) variables to provide a linearly comparable edge score.
7. Deduplicates results to only store unique non-directional combinations (i < j).
8. Combines integer data into binary numpy arrays and packs it structurally into HDF5 datasets.
"""
# %% Import Necessary Libraries
# Limit threads to prevent CPU thrashing
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import pickle
import subprocess
import shutil
import glob
import numpy as np
import math
import multiprocessing
import json
import h5py
from Bio import SeqIO
from tqdm import tqdm

# %% =======================================
# CONFIGURATION
# ==========================================

INPUT_FASTA = None

# SETTINGS
MATRIX = "BLOSUM62"
NUM_THREADS = 12
BATCH_SIZE = 500000
BLASTP_DIR = ""
SAFE_TEMP_DIR = os.path.join(os.path.expanduser("~"), "Alignment_TEMP")

# EXECUTABLE PATHS (Will be resolved dynamically after settings load)
NCBI_BIN_DIR    = r"C:\Program Files\NCBI"
MAKEBLASTDB_CMD = "makeblastdb"
BLASTP_CMD      = "blastp"

FASTA_DIR = os.path.join("..", "Input_Files", "Sequence_Sets")
NETWORK_DIR = os.path.join("..", "Input_Files", "Networks_EValues")

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

# Resolve BLAST commands after config overrides

# Resolve BLASTP and MAKEBLASTDB paths
if BLASTP_DIR and os.path.exists(BLASTP_DIR):
    MAKEBLASTDB_CMD = os.path.join(BLASTP_DIR, "makeblastdb.exe" if os.name == "nt" else "makeblastdb")
    BLASTP_CMD = os.path.join(BLASTP_DIR, "blastp.exe" if os.name == "nt" else "blastp")
else:
    # Fallback search if not specified or empty
    if shutil.which("blastp") or shutil.which("blastp.exe"):
        MAKEBLASTDB_CMD = "makeblastdb"
        BLASTP_CMD = "blastp"
    else:
        # Check standard default installation folders depending on OS
        if os.name == "nt":
            # Search C:\Program Files\NCBI dynamically for any version of BLAST
            ncbi_dir = r"C:\Program Files\NCBI"
            found_dir = None
            if os.path.exists(ncbi_dir):
                try:
                    valid_dirs = []
                    for d in os.listdir(ncbi_dir):
                        bin_path = os.path.join(ncbi_dir, d, "bin")
                        if os.path.exists(os.path.join(bin_path, "blastp.exe")):
                            valid_dirs.append(bin_path)
                    if valid_dirs:
                        valid_dirs.sort(reverse=True)
                        found_dir = valid_dirs[0]
                except:
                    pass
            
            if found_dir:
                MAKEBLASTDB_CMD = os.path.join(found_dir, "makeblastdb.exe")
                BLASTP_CMD = os.path.join(found_dir, "blastp.exe")
            else:
                MAKEBLASTDB_CMD = "makeblastdb"
                BLASTP_CMD = "blastp"
        else:
            # Unix fallback search
            unix_fallbacks = [
                "/usr/local/ncbi/blast/bin",
                "/usr/local/bin",
                "/usr/bin",
                "/opt/homebrew/bin"
            ]
            found_dir = None
            for path in unix_fallbacks:
                if os.path.exists(os.path.join(path, "blastp")):
                    found_dir = path
                    break
            
            if found_dir:
                MAKEBLASTDB_CMD = os.path.join(found_dir, "makeblastdb")
                BLASTP_CMD = os.path.join(found_dir, "blastp")
            else:
                MAKEBLASTDB_CMD = "makeblastdb"
                BLASTP_CMD = "blastp"

# ADVANCED
E_VALUE_CUTOFF = 1e300 # Maximum E-value threshold; sequence hit pairs evaluated above this cutoff are entirely discarded.
MAX_TARGET_SEQS = 1000000 # The maximum threshold of mathematically aligned sequence hit traces retained per query.

SEQUENCE_SET = INPUT_FASTA.replace(".fasta", "")
SAFE_TEMP_DIR = os.path.join(SAFE_TEMP_DIR, SEQUENCE_SET)

FULL_INPUT_FASTA = os.path.join(FASTA_DIR, INPUT_FASTA)
OUTPUT_HDF5 = os.path.join(NETWORK_DIR, f"{SEQUENCE_SET}_[BLAST]_EValue.h5")

# WORKSPACE SETUP
CHUNKS_DIR    = os.path.join(SAFE_TEMP_DIR, "chunks")
RESULTS_DIR   = os.path.join(SAFE_TEMP_DIR, "results")
BATCH_DIR     = os.path.join(SAFE_TEMP_DIR, "batches")
CONFIG_FILE   = os.path.join(SAFE_TEMP_DIR, "job_config.json")

# %% =======================================
# HELPER FUNCTIONS
# ==========================================

def check_and_initialize_workspace():
    current_config = {
        "sequence_set": SEQUENCE_SET,
        "matrix": MATRIX,
        "e_value": E_VALUE_CUTOFF,
        "input_file": FULL_INPUT_FASTA
    }
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f: saved_config = json.load(f)
            if saved_config != current_config:
                print("\n⚠️  Configuration changed! Wiping temp folder...")
                shutil.rmtree(SAFE_TEMP_DIR)
            else:
                print("\n✅ Resuming previous run...")
        except: shutil.rmtree(SAFE_TEMP_DIR)
            
    for d in [SAFE_TEMP_DIR, CHUNKS_DIR, RESULTS_DIR, BATCH_DIR]:
        os.makedirs(d, exist_ok=True)
        
    with open(CONFIG_FILE, "w") as f: json.dump(current_config, f, indent=4)

def save_batch(data, batch_id):
    filename = os.path.join(BATCH_DIR, f"batch_{batch_id:05d}.pkl")
    with open(filename, "wb") as f: pickle.dump(data, f)

def split_fasta_into_chunks(fasta_path, num_chunks, output_dir):
    existing_chunks = glob.glob(os.path.join(output_dir, "chunk_*.fasta"))
    if len(existing_chunks) == num_chunks:
        print("-> Using existing query chunks.")
        return sorted(existing_chunks)
    
    print("-> Splitting query file...")
    for f in existing_chunks: os.remove(f)
    
    records = list(SeqIO.parse(fasta_path, "fasta"))
    chunk_size = math.ceil(len(records) / num_chunks)
    chunk_files = []
    
    for i in range(num_chunks):
        chunk_records = records[i*chunk_size : (i+1)*chunk_size]
        if not chunk_records: continue
        chunk_name = f"chunk_{i}.fasta"
        chunk_path = os.path.join(output_dir, chunk_name)
        SeqIO.write(chunk_records, chunk_path, "fasta")
        chunk_files.append(chunk_path)
        
    return chunk_files

def run_alignment_worker(args):
    """
    Runs the BLAST command.
    """
    query_file, target_ref, exe_path, matrix, evalue, out_file = args
    if os.path.exists(out_file) and os.path.getsize(out_file) > 0: return "Skipped"

    limit_count = str(MAX_TARGET_SEQS) 

    cmd = [
        exe_path, "-query", query_file, "-db", target_ref, "-out", out_file,
        "-outfmt", "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore",
        "-matrix", matrix, "-evalue", str(evalue),
        "-max_target_seqs", limit_count, "-max_hsps", "1", "-comp_based_stats", "0" 
    ]
    
    try:
        # Capture stderr for debugging
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        return "Done"
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"
    except Exception as e:
        return f"SysError: {str(e)}"

# %% =======================================
# MAIN WORKFLOW
# ==========================================

def run_workflow():
    print(f"--- BLAST All-vs-All (-Log10 E-Value Mode) ---")
    
    check_and_initialize_workspace()
    
    # 0. RESOLVE INPUT PATH
    clean_fasta_path = os.path.normpath(FULL_INPUT_FASTA)
    if not os.path.exists(clean_fasta_path):
        print(f"❌ Error: Input FASTA not found at:\n{clean_fasta_path}")
        sys.exit(1)

    # 1. READ HEADERS & GENERATE SAFE FASTA
    print(f"Reading headers from {clean_fasta_path} and generating safe IDs...")
    headers = []
    
    # Create a safe working FASTA where IDs are guaranteed unique integers
    safe_fasta_path = os.path.join(SAFE_TEMP_DIR, f"{SEQUENCE_SET}_safe.fasta")
    
    with open(safe_fasta_path, "w") as out_fasta:
        for i, record in enumerate(SeqIO.parse(clean_fasta_path, "fasta")):
            # Store the exact, unadulterated header
            headers.append(record.description) 
            # Write a sanitized sequence for BLAST using the index as the ID
            out_fasta.write(f">{i}\n{str(record.seq)}\n")

    # 2. PREPARE TARGET DB (BLAST Only)
    db_name = os.path.join(SAFE_TEMP_DIR, "temp_db")
    if not os.path.exists(db_name + ".pin"):
        print("Building BLAST Database...")
        # Point directly to the newly generated safe FASTA
        cmd_db = [MAKEBLASTDB_CMD, "-in", safe_fasta_path, "-dbtype", "prot", "-out", db_name]
        
        try:
            subprocess.run(cmd_db, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print("❌ BLAST DATABASE ERROR\nSTDOUT:", e.stdout, "\nSTDERR:", e.stderr); sys.exit(1)
        
    target_reference = db_name

    # 3. SPLIT & RUN
    chunks = split_fasta_into_chunks(safe_fasta_path, NUM_THREADS, CHUNKS_DIR)
    print(f"Running {len(chunks)} alignment jobs...")
    
    tasks = []
    output_files = []
    for i, chunk in enumerate(chunks):
        out_file = os.path.join(RESULTS_DIR, f"result_{i}.txt")
        output_files.append(out_file)
        tasks.append((chunk, target_reference, BLASTP_CMD, MATRIX, E_VALUE_CUTOFF, out_file))
    
    # Run and check for errors
    error_log = []
    with multiprocessing.Pool(processes=NUM_THREADS) as pool:
        for result in tqdm(pool.imap(run_alignment_worker, tasks), total=len(tasks), desc="Aligning"):
            if result.startswith("Error") or result.startswith("SysError"):
                error_log.append(result)

    if error_log:
        print("\n❌ CRITICAL: BLAST Worker Failed!")
        print("Last Error Message:")
        print(error_log[-1])
        sys.exit(1)

    # 5. PARSE & CONVERT
    print("\n" + "="*40)
    print("PARSING RESULTS & CHECKING HEADERS")
    print("="*40)
    
    # Debug Counters
    total_lines = 0
    match_count = 0
    mismatch_count = 0
    file_count = 0
    debug_printed = False

    # Clear old batches
    existing_batches = glob.glob(os.path.join(BATCH_DIR, "batch_*.pkl"))
    for b in existing_batches:
        try: os.remove(b)
        except: pass

    batch_src, batch_tgt, batch_scr = [], [], []
    batch_id = 0
    
    for out_file in tqdm(output_files, desc="Parsing"):
        if not os.path.exists(out_file): continue
        if os.path.getsize(out_file) == 0: continue

        file_count += 1
        
        with open(out_file, "r") as f:
            for line in f:
                # 1. Clean Line
                line = line.strip()
                if not line: continue
                if line.startswith("#"): continue
                
                cols = line.split()
                # BLAST fmt 6 with our string should have 12 columns
                if len(cols) < 11: continue
                
                # 2. Extract Fields (Cols 0, 1, 10 are Query, Subject, E-value)
                q_id, s_id, eval_str = cols[0], cols[1], cols[10]
                
                # Verify E-value is float
                try:
                    raw_e = float(eval_str)
                except ValueError:
                    continue

                # 3. Convert IDs directly to integers
                total_lines += 1
                
                try:
                    u = int(q_id)
                    v = int(s_id)
                except ValueError:
                    if not debug_printed:
                        print(f"\n❌ ID FORMAT ERROR!")
                        print(f"   Line content: '{line[:100]}...'")
                        print(f"   Expected an integer, got: '{q_id}'")
                        debug_printed = True
                    mismatch_count += 1
                    continue

                # Ensure the index actually exists in our headers list
                if u < len(headers) and v < len(headers):
                    match_count += 1
                    log_score = -math.log10(raw_e + 1e-300)
                    
                    if u < v: 
                        batch_src.append(u); batch_tgt.append(v); batch_scr.append(log_score)
                        if len(batch_src) >= BATCH_SIZE:
                            save_batch((batch_src, batch_tgt, batch_scr), batch_id)
                            batch_id += 1; batch_src, batch_tgt, batch_scr = [], [], []
                else:
                    mismatch_count += 1
                    
    if batch_src: save_batch((batch_src, batch_tgt, batch_scr), batch_id)

    print("\n" + "-"*40)
    print("PARSING DIAGNOSTICS")
    print("-"*40)
    print(f"Files Processed:  {file_count}")
    print(f"Total Valid Hits: {total_lines}")
    print(f"Matches Saved:    {match_count}")
    print(f"ID Mismatches:    {mismatch_count}")
    
    if total_lines == 0:
        print(">> WARNING: No alignments found. Check BLAST execution.")
    print("-"*40 + "\n")

    # 6. CONSOLIDATE
    print("Consolidating...")
    all_sources, all_targets, all_scores = [], [], []
    batch_pkls = sorted(glob.glob(os.path.join(BATCH_DIR, "batch_*.pkl")))
    
    if not batch_pkls:
        print("❌ Error: No valid data batches generated.")
        sys.exit(1)
    
    for pkl in tqdm(batch_pkls, desc="Merging"):
        with open(pkl, "rb") as f:
            data = pickle.load(f)
            if len(data) == 4: (b_src, b_tgt, b_scr, _) = data 
            else: (b_src, b_tgt, b_scr) = data
            all_sources.extend(b_src); all_targets.extend(b_tgt); all_scores.extend(b_scr)

    output_full_path = os.path.normpath(OUTPUT_HDF5)
    
    # 7. SAVE TO HDF5
    num_seqs = len(headers)
    idx_dtype = np.uint16 if num_seqs <= 65535 else np.uint32
    print(f"  > Selected {idx_dtype.__name__} for topology indices based on {num_seqs} sequences.")
    
    arr_i = np.array(all_sources, dtype=idx_dtype)
    arr_j = np.array(all_targets, dtype=idx_dtype)
    arr_score = np.array(all_scores, dtype=np.float32)

    # --- CHECK FINAL CONSISTENCY ---
    max_idx_used = 0
    if len(arr_i) > 0:
        max_idx_used = max(np.max(arr_i), np.max(arr_j))
    
    if max_idx_used >= len(headers):
         print(f"❌ FATAL ERROR: Index {max_idx_used} is out of bounds.")
         sys.exit(1)

    print("\n" + "="*40)
    print("DATA PREVIEW")
    print("="*40)
    print(f"Total Headers: {len(headers)}")
    print(f"Total Edges:   {len(arr_i)}")
    print("-" * 20)
    if len(arr_i) > 0:
        print(f"Headers [0]: {headers[0]}")
        print(f"Score   [0]: {arr_score[0]}")
    else:
        print("⚠️  NO DATA LOADED")
    print("="*40 + "\n")

    os.makedirs(os.path.dirname(output_full_path), exist_ok=True)
    
    print(f"Saving Combined Scores to {output_full_path}...")
    with h5py.File(output_full_path, "w") as hf:
        dt_str = h5py.string_dtype(encoding='utf-8')
        hf.create_dataset("headers", data=np.array(headers, dtype=object), dtype=dt_str)
        hf.create_dataset("i", data=arr_i)
        hf.create_dataset("j", data=arr_j)
        hf.create_dataset("score", data=arr_score)
        
    print(f"Done! Saved to {output_full_path}")
    print("\n🧹 Cleaning up temporary workspace...")
    try:
        shutil.rmtree(SAFE_TEMP_DIR)
        print(f"✅ Successfully deleted temporary folder: {SAFE_TEMP_DIR}")
    except Exception as e:
        print(f"⚠️ Could not automatically delete temp folder (files might be in use): {e}")

if __name__ == "__main__":
    run_workflow()