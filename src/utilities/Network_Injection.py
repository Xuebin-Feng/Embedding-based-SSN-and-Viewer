"""
File: Network_Injection.py
===================================
Description:
After adding new sequences to a precomputed embedding database (via `Embedding_Injection.py`), this script efficiently 
updates the massive all-vs-all similarity network without recalculating the pre-existing pairwise combinations.
It dynamically computes combinations involving the *new* sequences while rapidly copying existing connections from cache.

Input:
- The old existing HDF5 network containing pre-calculated alignment scores (`OLD_NETWORK`).
- The newly updated embeddings database containing both old and new tensors (`NEW_EMBEDDINGS`).
- (Optional) The old traceback paths file related to the network (`OLD_PATHS`).

Output:
- A new master HDF5 network containing all completed old and new edges (`FINAL_OUTPUT_NET`).
- (Optional) A synchronized updated traceback paths file (`FINAL_OUTPUT_PATHS`).

Settings:
- OLD_SEQUENCE_SET / NEW_SEQUENCE_SET: Filename parameters used to locate the input databases and save the updated output.
- MODEL_NAME: The model identifier matching the embeddings used.
- BATCH_SIZE: Number of pairs to process between writing to intermediate temp files (RAM protection).
- WORKERS: Threads used for multiprocessing new alignment combinations.

Algorithm:
1. Loads the headers of both the OLD network and the NEW embedding database.
2. Mathematically calculates the exact number of pairs required for an all-vs-all grid `(N*(N-1))/2` for the new sequence length.
3. Pre-allocates massive, correctly sized blank contiguous 1D numpy arrays into system memory.
4. Iterates linearly over every theoretical combination pair in the new index space.
5. If both headers previously existed, it converts their 2D coordinates back into a 1D flat index and instantly copies 
   the raw alignment score bytes straight into the new arrays in RAM.
6. If the combination involves a new header, the required index coordinates are tossed into a "To-Do" queue.
7. Multiprocessing CPU workers iterate over the to-do queue, plucking the required embeddings from disk and directly evaluating their Needleman-Wunsch/Smith-Waterman scores.
8. Calculated results are merged into the massive contiguous array space and structurally streamed back to disk as HDF5 arrays.
"""
# %% Import Necessary Libraries
# Limit threads to prevent CPU thrashing
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import zlib
import pickle
import numpy as np
import h5py
import torch
import glob
import math
import shutil
import sys
import hashlib
from multiprocessing import Pool, set_start_method
from tqdm import tqdm
from numba import jit
import Hardware_Utils

# ==========================================
# CONFIGURATION
# ==========================================
# INPUTS (Filenames only, provided by GUI)
OLD_NETWORK = None 
NEW_EMBEDDINGS = None 

# SETTINGS
BATCH_SIZE = 500000 
WORKERS = 12 
LOCAL_GAP_P = -2.0 
GLOBAL_GAP_P = 0.0 

# DIRECTORIES
NETWORK_DIR = os.path.join("..", "Input_Files", "Networks_EValues")
EMBED_DIR = os.path.join("..", "Embeddings")
PATH_DIR = os.path.join("..", "Cache_Files", "Global_Path")

# --- JSON Settings Override ---
import json
import ast

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

# --- DYNAMIC INFERENCE ---
import re

# 1. Dynamically parse names from the input files provided by the GUI
_old_seq_set = "UnknownOld"
_model_name = "UnknownModel"
match_old = re.search(r"^(.*)_\[(.*)\]_network\.h5$", OLD_NETWORK)
if match_old:
    _old_seq_set = match_old.group(1)
    _model_name = match_old.group(2)

_new_seq_set = "UnknownNew"
match_new = re.search(r"^(.*)_\[(.*)\]_embeddings\.h5$", NEW_EMBEDDINGS)
if match_new:
    _new_seq_set = match_new.group(1)

# 2. Build Full Input Paths
OLD_NETWORK = os.path.join(NETWORK_DIR, OLD_NETWORK) if NETWORK_DIR else ""
NEW_EMBEDDINGS = os.path.join(EMBED_DIR, NEW_EMBEDDINGS) if EMBED_DIR else ""
OLD_PATHS = os.path.join(PATH_DIR, f"{_old_seq_set}_[{_model_name}]_paths.h5") if PATH_DIR else ""

# 3. Build Full Output Paths
RESULTS_DIR = os.path.join(NETWORK_DIR, f"{_new_seq_set}_[{_model_name}]_network_temp") if NETWORK_DIR else ""
FINAL_OUTPUT_NET = os.path.join(NETWORK_DIR, f"{_new_seq_set}_[{_model_name}]_network.h5") if NETWORK_DIR else ""
FINAL_OUTPUT_PATHS = os.path.join(PATH_DIR, f"{_new_seq_set}_[{_model_name}]_paths.h5") if PATH_DIR else ""
CONFIG_FILE = os.path.join(RESULTS_DIR, "injection_config.json")

# %% =======================================
# KERNELS 
# ==========================================
def calculate_file_hash(file_path):
    """
    Computes the MD5 checksum of a file in chunks to verify consistency.
    """
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_score_matrix_torch(emb_i, emb_j, device):
    t_i = torch.as_tensor(emb_i, device=device, dtype=torch.float16)
    t_j = torch.as_tensor(emb_j, device=device, dtype=torch.float16)
    dist_mat = torch.cdist(t_i, t_j, p=2)
    sim_mat = torch.exp(-dist_mat)
    
    epsilon = 1e-8
    row_mean = sim_mat.mean(dim=1, keepdim=True)
    row_std = sim_mat.std(dim=1, keepdim=True)
    col_mean = sim_mat.mean(dim=0, keepdim=True)
    col_std = sim_mat.std(dim=0, keepdim=True)
    
    z_r = (sim_mat - row_mean) / (row_std + epsilon)
    z_c = (sim_mat - col_mean) / (col_std + epsilon)
    final_score = (z_r + z_c) / 2.0
    
    return final_score.to(dtype=torch.float32, device="cpu").numpy()

@jit(nopython=True, fastmath=True)
def run_global_traceback(score_matrix, gap_p):
    N, M = score_matrix.shape
    dp = np.zeros((N + 1, M + 1), dtype=np.float32)
    pointer = np.zeros((N + 1, M + 1), dtype=np.int8)

    for c in range(1, M + 1): dp[0, c] = c * gap_p; pointer[0, c] = 3
    for r in range(1, N + 1): dp[r, 0] = r * gap_p; pointer[r, 0] = 2
    pointer[0, 0] = 0

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            match = dp[i-1, j-1] + score_matrix[i-1, j-1]
            delete = dp[i-1, j] + gap_p
            insert = dp[i, j-1] + gap_p
            
            best = match; ptr = 1
            if delete > best: best = delete; ptr = 2
            if insert > best: best = insert; ptr = 3
            
            dp[i, j] = best
            pointer[i, j] = ptr

    max_score = dp[N, M]
    path_buffer = np.zeros(N + M, dtype=np.int8)
    k = 0
    i, j = N, M
    while i > 0 or j > 0:
        p = pointer[i, j]
        path_buffer[k] = p
        k += 1
        if p == 1: i -= 1; j -= 1
        elif p == 2: i -= 1
        elif p == 3: j -= 1
        else: break 

    return max_score, k, path_buffer[:k]

@jit(nopython=True, fastmath=True)
def run_local_traceback(score_matrix, gap_p):
    N, M = score_matrix.shape
    dp = np.zeros((N + 1, M + 1), dtype=np.float32)
    pointer = np.zeros((N + 1, M + 1), dtype=np.int8)
    
    max_score = 0.0
    max_i, max_j = 0, 0
    
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            match = dp[i-1, j-1] + score_matrix[i-1, j-1]
            delete = dp[i-1, j] + gap_p
            insert = dp[i, j-1] + gap_p
            
            best = 0.0
            ptr = 0
            
            if match > best: best = match; ptr = 1
            if delete > best: best = delete; ptr = 2
            if insert > best: best = insert; ptr = 3
            
            dp[i, j] = best
            pointer[i, j] = ptr
            
            if best > max_score:
                max_score = best
                max_i = i
                max_j = j
                
    align_len = 0
    curr_i, curr_j = max_i, max_j
    while curr_i > 0 and curr_j > 0:
        if dp[curr_i, curr_j] == 0: break 
        p = pointer[curr_i, curr_j]
        if p == 0: break
        align_len += 1
        if p == 1: curr_i -= 1; curr_j -= 1
        elif p == 2: curr_i -= 1
        elif p == 3: curr_j -= 1
        else: break
            
    return max_score, align_len

@jit(nopython=True)
def pack_path_2bit(path_arr):
    n = len(path_arr)
    n_bytes = (n + 3) // 4
    out = np.zeros(n_bytes, dtype=np.uint8)
    for i in range(n_bytes):
        val = 0
        for b in range(4):
            idx = i * 4 + b
            if idx < n:
                val |= (path_arr[idx] << (b * 2))
        out[i] = val
    return out


# %% =======================================
# HDF5 WORKER INITIALIZATION
# ==========================================
worker_hf = None

def init_worker(h5_path):
    global worker_hf
    worker_hf = h5py.File(h5_path, "r", libver='latest', swmr=True)


# %% =======================================
# HYBRID WORKER & RESUME LOGIC
# ==========================================
def calculate_hybrid_data(args):
    idx_i, idx_j, safe_h_i, safe_h_j = args
    global worker_hf
    
    device = Hardware_Utils.get_optimal_device()
    
    emb_i = worker_hf["embeddings"][safe_h_i][:]
    emb_j = worker_hf["embeddings"][safe_h_j][:]
    
    matrix = compute_score_matrix_torch(emb_i, emb_j, device)
    
    g_raw, g_len, path_array = run_global_traceback(matrix, GLOBAL_GAP_P)
    packed_path = pack_path_2bit(path_array)
    c_path = zlib.compress(packed_path.tobytes())
    
    matrix -= 2.0 
    l_raw, l_len = run_local_traceback(matrix, LOCAL_GAP_P)

    return (idx_i, idx_j, l_raw, l_len, g_raw, g_len, c_path)

def process_batch(batch_tasks, batch_id, workers, new_emb_path, has_old_paths, embedding_checksum):
    output_filename = os.path.join(RESULTS_DIR, f"batch_{batch_id:05d}.h5")
    results = []
    with Pool(processes=workers, initializer=init_worker, initargs=(new_emb_path,)) as pool:
        iterator = pool.imap_unordered(calculate_hybrid_data, batch_tasks, chunksize=10)
        for res in tqdm(iterator, total=len(batch_tasks), desc=f"  Batch {batch_id}", leave=False):
            results.append(res)
            
    # Save as uncompressed HDF5 for easy access
    rows_i = [r[0] for r in results]
    rows_j = [r[1] for r in results]
    l_scores = [r[2] for r in results]
    l_lens = [r[3] for r in results]
    g_scores = [r[4] for r in results]
    g_lens = [r[5] for r in results]
    
    with h5py.File(output_filename, "w") as hf:
        if embedding_checksum is not None:
            hf.attrs["embedding_checksum"] = embedding_checksum
            
        hf.create_dataset("i", data=np.array(rows_i, dtype=np.uint32))
        hf.create_dataset("j", data=np.array(rows_j, dtype=np.uint32))
        hf.create_dataset("l_score", data=np.array(l_scores, dtype=np.float32))
        hf.create_dataset("l_len", data=np.array(l_lens, dtype=np.uint16))
        hf.create_dataset("g_score", data=np.array(g_scores, dtype=np.float32))
        hf.create_dataset("g_len", data=np.array(g_lens, dtype=np.uint16))
        if has_old_paths:
            edges_paths = [r[6] for r in results]
            dt_vlen = h5py.vlen_dtype(np.uint8)
            np_paths = np.empty(len(edges_paths), dtype=object)
            for idx, p in enumerate(edges_paths):
                np_paths[idx] = np.frombuffer(p, dtype=np.uint8)
            hf.create_dataset("paths", data=np_paths, dtype=dt_vlen)

def scan_existing_batches(new_N, current_checksum):
    computed_pairs = set()
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        return computed_pairs
    
    batch_files = glob.glob(os.path.join(glob.escape(RESULTS_DIR), "batch_*.h5"))
    for bf in batch_files:
        try:
            with h5py.File(bf, "r") as hf:
                cached_checksum = hf.attrs.get("embedding_checksum")
                if cached_checksum != current_checksum:
                    print(f"⚠️ Checksum mismatch in {bf}! Deleting mismatched cache file...")
                    hf.close()
                    try:
                        os.remove(bf)
                    except Exception as err:
                        print(f"Error removing mismatched file {bf}: {err}")
                    continue
                
                if "i" in hf and "j" in hf:
                    arr_i = hf["i"][:]
                    arr_j = hf["j"][:]
                    for i_val, j_val in zip(arr_i, arr_j):
                        computed_pairs.add(min(i_val, j_val) * new_N + max(i_val, j_val))
        except Exception as e:
            print(f"Warning: Could not read batch file {bf}: {e}")
    return computed_pairs

def compile_final_output(new_headers, seq_lens, required_pairs, cached_old_pairs, new_N, 
                         old_header_to_idx, old_l_score, old_l_len, old_g_score, old_g_len, 
                         has_old_paths, OLD_PATHS, actual_idx_map):
    print(f"\n--- Compiling Final HDF5 Output (Keep count: {len(required_pairs)}) ---")
    
    num_kept = len(required_pairs)
    idx_dtype = np.uint16 if new_N <= 65535 else np.uint32
    
    final_i = np.zeros(num_kept, dtype=idx_dtype)
    final_j = np.zeros(num_kept, dtype=idx_dtype)
    final_l_score = np.zeros(num_kept, dtype=np.float32)
    final_l_len = np.zeros(num_kept, dtype=np.uint16)
    final_g_score = np.zeros(num_kept, dtype=np.float32)
    final_g_len = np.zeros(num_kept, dtype=np.uint16)
    final_paths = [None] * num_kept if has_old_paths else None
    
    # 1. Load old paths into memory if needed
    old_paths_mem = None
    if has_old_paths:
        try:
            with h5py.File(OLD_PATHS, "r") as hf_old_paths:
                old_paths_mem = hf_old_paths['paths'][:]
        except Exception as e:
            print(f"Warning: Could not read old paths from {OLD_PATHS}: {e}")
            has_old_paths = False
            final_paths = None
            
    # 2. Fill final arrays with pre-existing cached old network data
    sorted_pairs = sorted(list(required_pairs))
    
    # Build mapping of pair_id to index in the final arrays
    pair_id_to_final_idx = {}
    for idx, pair_id in enumerate(sorted_pairs):
        i = pair_id // new_N
        j = pair_id % new_N
        final_i[idx] = i
        final_j[idx] = j
        pair_id_to_final_idx[pair_id] = idx
        
    # Copy cached old network data
    old_headers_set = set(old_header_to_idx.keys())
    for pair_id in cached_old_pairs:
        if pair_id in pair_id_to_final_idx:
            idx = pair_id_to_final_idx[pair_id]
            i = pair_id // new_N
            j = pair_id % new_N
            h_i = new_headers[i]
            h_j = new_headers[j]
            u = old_header_to_idx[h_i]
            v = old_header_to_idx[h_j]
            # Retrieve theoretical flat index of the old network
            theoretical_idx = int(u * len(old_headers_set) - u * (u + 1) // 2 + (v - u - 1)) if u < v else int(v * len(old_headers_set) - v * (v + 1) // 2 + (u - v - 1))
            actual_idx = actual_idx_map[theoretical_idx]
            
            final_l_score[idx] = old_l_score[actual_idx]
            final_l_len[idx] = old_l_len[actual_idx]
            final_g_score[idx] = old_g_score[actual_idx]
            final_g_len[idx] = old_g_len[actual_idx]
            if has_old_paths and old_paths_mem is not None:
                final_paths[idx] = old_paths_mem[actual_idx]
                
    # 3. Read calculated batches and populate remaining entries
    batch_files = sorted(glob.glob(os.path.join(glob.escape(RESULTS_DIR), "batch_*.h5")))
    for bf in tqdm(batch_files, desc="Filtering computed batches"):
        try:
            with h5py.File(bf, "r") as hf:
                if "i" not in hf or "j" not in hf:
                    continue
                arr_i = hf["i"][:]
                arr_j = hf["j"][:]
                arr_l_score = hf["l_score"][:]
                arr_l_len = hf["l_len"][:]
                arr_g_score = hf["g_score"][:]
                arr_g_len = hf["g_len"][:]
                
                arr_paths = None
                if "paths" in hf:
                    arr_paths = hf["paths"][:]
                    
                for k in range(len(arr_i)):
                    i_val = arr_i[k]
                    j_val = arr_j[k]
                    pair_id = min(i_val, j_val) * new_N + max(i_val, j_val)
                    # Copy from batch files ONLY if it is a required pair AND was not copied from cache
                    if pair_id in pair_id_to_final_idx and pair_id not in cached_old_pairs:
                        idx = pair_id_to_final_idx[pair_id]
                        final_l_score[idx] = arr_l_score[k]
                        final_l_len[idx] = arr_l_len[k]
                        final_g_score[idx] = arr_g_score[k]
                        final_g_len[idx] = arr_g_len[k]
                        if has_old_paths and arr_paths is not None:
                            final_paths[idx] = arr_paths[k]
        except Exception as e:
            print(f"Warning: Error reading {bf} during compilation: {e}")
            
    # 4. Save to Final network HDF5
    print(f"Saving Combined Scores to {FINAL_OUTPUT_NET}...")
    os.makedirs(os.path.dirname(FINAL_OUTPUT_NET), exist_ok=True)
    with h5py.File(FINAL_OUTPUT_NET, "w") as hf_out:
        dt_str = h5py.string_dtype(encoding='utf-8')
        hf_out.create_dataset("headers", data=np.array(new_headers, dtype=object), dtype=dt_str)
        hf_out.create_dataset("seq_lens", data=np.array(seq_lens, dtype=np.uint16))
        
        # Save arrays
        hf_out.create_dataset("i", data=final_i)
        hf_out.create_dataset("j", data=final_j)
        hf_out.create_dataset("l_score", data=final_l_score)
        hf_out.create_dataset("l_len", data=final_l_len)
        hf_out.create_dataset("g_score", data=final_g_score)
        hf_out.create_dataset("g_len", data=final_g_len)
        
    if has_old_paths:
        print(f"Saving Global Traceback Paths to {FINAL_OUTPUT_PATHS}...")
        os.makedirs(os.path.dirname(FINAL_OUTPUT_PATHS), exist_ok=True)
        with h5py.File(FINAL_OUTPUT_PATHS, "w") as hf_p_out:
            dt_str = h5py.string_dtype(encoding='utf-8')
            hf_p_out.create_dataset("headers", data=np.array(new_headers, dtype=object), dtype=dt_str)
            
            dt_vlen = h5py.vlen_dtype(np.uint8)
            print("Formatting paths for bulk HDF5 write...")
            np_paths = np.empty(len(final_paths), dtype=object)
            for idx, p in enumerate(final_paths):
                if p is None:
                    np_paths[idx] = np.array([], dtype=np.uint8)
                else:
                    np_paths[idx] = np.frombuffer(p, dtype=np.uint8)
                    
            hf_p_out.create_dataset("paths", data=np_paths, dtype=dt_vlen)
            
    print("✅ Compilation complete!")


# %% =======================================
# MAIN INJECTION LOGIC
# ==========================================
def run_injection():
    try: set_start_method('spawn')
    except RuntimeError: pass

    # 1. Load Data
    print("Loading New Embeddings Metadata...")
    if not os.path.exists(NEW_EMBEDDINGS):
        sys.exit(f"❌ Error: New embeddings file not found at {NEW_EMBEDDINGS}")
        
    # Calculate checksum of the input embedding file to validate cache
    print("Calculating checksum of input embedding file...")
    current_checksum = calculate_file_hash(NEW_EMBEDDINGS)
    print(f"  > Checksum: {current_checksum}")

    with h5py.File(NEW_EMBEDDINGS, "r") as hf_new:
        raw_new_headers = hf_new["headers"][:]
        new_headers = [h.decode('utf-8') if isinstance(h, bytes) else h for h in raw_new_headers]
        new_N = len(new_headers)
        
        seq_lens = []
        safe_new_headers = []
        for h in new_headers:
            safe_h = h.replace("/", "_").replace("\\", "_")
            safe_new_headers.append(safe_h)
            seq_lens.append(hf_new["embeddings"][safe_h].shape[0])
            
    # Initialize workspace folder without wipes
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading Existing Network...")
    if not os.path.exists(OLD_NETWORK):
        sys.exit(f"❌ Error: Old network file not found at {OLD_NETWORK}")
        
    with h5py.File(OLD_NETWORK, "r") as hf_old_net:
        raw_old_headers = hf_old_net['headers'][:]
        old_headers = [h.decode('utf-8') if isinstance(h, bytes) else h for h in raw_old_headers]
        old_N = len(old_headers)
        old_header_to_idx = {h: i for i, h in enumerate(old_headers)}
        
        # ---> FIX: Load original coordinates to untangle the unordered cache
        old_i = hf_old_net['i'][:]
        old_j = hf_old_net['j'][:]
        
        old_l_score = hf_old_net['l_score'][:]
        old_l_len = hf_old_net['l_len'][:]
        old_g_score = hf_old_net['g_score'][:]
        old_g_len = hf_old_net['g_len'][:]

    # ---> FIX: Build a reverse-lookup array (Vectorized for extreme speed)
    print("Building reverse-lookup map for unordered cache...")
    total_old_pairs = len(old_i)
    theoretical_old_total = (old_N * (old_N - 1)) // 2
    
    # Clean fix for sparse index out of bounds: size must be theoretical_old_total
    actual_idx_map = np.zeros(theoretical_old_total, dtype=np.uint32) 
    
    old_i_64 = old_i.astype(np.int64)
    old_j_64 = old_j.astype(np.int64)
    theoretical_flat_indices = (old_i_64 * old_N) - (old_i_64 * (old_i_64 + 1) // 2) + (old_j_64 - old_i_64 - 1)
    
    actual_idx_map[theoretical_flat_indices] = np.arange(total_old_pairs)
    
    exists_in_old = np.zeros(theoretical_old_total, dtype=bool)
    exists_in_old[theoretical_flat_indices] = True

    # 2. Check for sparse old network and precalculate similarity threshold
    is_sparse_old = total_old_pairs < theoretical_old_total
    cos_sim = None
    threshold_cos = -1.0
    
    if is_sparse_old:
        percent_exists = (total_old_pairs / theoretical_old_total) * 100
        print(f"\n--- 🛠️ Sparse Input Network Detected: {total_old_pairs} / {theoretical_old_total} edges exist ({percent_exists:.2f}%) ---")
        
        # Compute mean embeddings for the new headers
        print("Loading mean embeddings for new sequence set...")
        mean_embs = []
        with h5py.File(NEW_EMBEDDINGS, "r") as hf_new_emb:
            for h in tqdm(safe_new_headers, desc="Calculating mean embeddings"):
                emb = hf_new_emb["embeddings"][h][:]
                mean_emb = np.mean(emb, axis=0)
                mean_embs.append(mean_emb)
        mean_embs = np.array(mean_embs, dtype=np.float32)
        
        # Calculate all-vs-all cosine similarities on CPU
        norms = np.linalg.norm(mean_embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)
        norm_embs = mean_embs / norms
        print("Computing all-vs-all cosine similarities...")
        cos_sim = np.dot(norm_embs, norm_embs.T)
        
        # Map headers to new indices
        new_header_to_idx = {h: idx for idx, h in enumerate(new_headers)}
        
        # Find the lowest cosine similarity among all sequence pairs (edges) present in the old network
        if total_old_pairs > 0:
            min_cos_val = 1.0
            min_cos_k = -1
            
            for k in range(total_old_pairs):
                u_h = old_headers[old_i[k]]
                v_h = old_headers[old_j[k]]
                if u_h in new_header_to_idx and v_h in new_header_to_idx:
                    u_new_idx = new_header_to_idx[u_h]
                    v_new_idx = new_header_to_idx[v_h]
                    sim_val = float(cos_sim[u_new_idx, v_new_idx])
                    if sim_val < min_cos_val:
                        min_cos_val = sim_val
                        min_cos_k = k
            
            if min_cos_k != -1:
                threshold_cos = min_cos_val
                h_i_min = old_headers[old_i[min_cos_k]]
                h_j_min = old_headers[old_j[min_cos_k]]
                print(f"  > Lowest cosine similarity in input network: {threshold_cos:.6f} (between '{h_i_min}' and '{h_j_min}', score: {old_l_score[min_cos_k]:.4f})")
            else:
                threshold_cos = 0.0
                print("  > Warning: No old network edges could be matched in new headers.")
                
            print(f"  > Setting cosine similarity prefiltering threshold to: {threshold_cos:.6f}")
            
            # Report any interaction within the original network that has lower cosine similarity than threshold_cos
            print("  > Scanning original network for edges below the threshold...")
            reported_count = 0
            for k in range(total_old_pairs):
                u_h = old_headers[old_i[k]]
                v_h = old_headers[old_j[k]]
                if u_h in new_header_to_idx and v_h in new_header_to_idx:
                    u_new_idx = new_header_to_idx[u_h]
                    v_new_idx = new_header_to_idx[v_h]
                    sim_val = cos_sim[u_new_idx, v_new_idx]
                    if sim_val < threshold_cos - 1e-5: # Floating point tolerance
                        if reported_count < 10:
                            print(f"    * Warning: Edge '{u_h}' - '{v_h}' has cosine similarity {sim_val:.6f} < threshold {threshold_cos:.6f} (score: {old_l_score[k]:.4f})")
                        reported_count += 1
            if reported_count > 0:
                print(f"  > Total edges in original network below threshold: {reported_count}")
            else:
                print("  > No edges in original network are below the threshold. (As expected!)")
        else:
            print("  > Warning: Old network is empty.")
        print("---------------------------------------------------------------------------\n")

    has_old_paths = os.path.exists(OLD_PATHS)
    if not has_old_paths:
        print(f"\n⚠️  Warning: Old paths file not found at {OLD_PATHS}.")
        print("   Proceeding with network update only. New paths will not be saved.\n")

    total_new_pairs = (new_N * (new_N - 1)) // 2
    
    # 3. Determine which edges are kept and count them (Job Manager required_pairs)
    old_headers_set = set(old_headers)
    
    # Pre-build cached_old_pairs flat index set using new indices
    cached_old_pairs = set()
    old_header_to_new_idx = {h: idx for idx, h in enumerate(new_headers)}
    for k in range(total_old_pairs):
        u_h = old_headers[old_i[k]]
        v_h = old_headers[old_j[k]]
        if u_h in old_header_to_new_idx and v_h in old_header_to_new_idx:
            u_idx = old_header_to_new_idx[u_h]
            v_idx = old_header_to_new_idx[v_h]
            cached_old_pairs.add(min(u_idx, v_idx) * new_N + max(u_idx, v_idx))
            
    # Establish required_pairs
    required_pairs = set()
    for i in range(new_N):
        h_i = new_headers[i]
        is_old_i = h_i in old_headers_set
        for j in range(i + 1, new_N):
            h_j = new_headers[j]
            is_old_j = h_j in old_headers_set
            
            keep = False
            if is_old_i and is_old_j:
                pair_id = i * new_N + j
                if pair_id in cached_old_pairs:
                    keep = True
            else:
                if not is_sparse_old or (cos_sim is not None and cos_sim[i, j] >= threshold_cos):
                    keep = True
                    
            if keep:
                required_pairs.add(i * new_N + j)
                
    effective_total_pairs = len(required_pairs)

    # 4. Scan existing batch files for already computed pairs
    computed_pairs = scan_existing_batches(new_N, current_checksum)

    # Find the next available batch_id
    existing_batches = glob.glob(os.path.join(glob.escape(RESULTS_DIR), "batch_*.h5"))
    batch_ids = []
    for f in existing_batches:
        base = os.path.basename(f)
        try:
            num = int(base.split("_")[1].split(".")[0])
            batch_ids.append(num)
        except:
            pass
    next_batch_id = max(batch_ids) + 1 if batch_ids else 0

    # Build queue of missing tasks to compute
    # Tasks should not be cached in the old network AND should not be computed yet
    tasks_to_compute = []
    for i in range(new_N):
        h_i = new_headers[i]
        for j in range(i + 1, new_N):
            pair_id = i * new_N + j
            if pair_id in required_pairs and pair_id not in cached_old_pairs and pair_id not in computed_pairs:
                tasks_to_compute.append((i, j, safe_new_headers[i], safe_new_headers[j]))

    num_tasks = len(tasks_to_compute)
    print(f"Total Required pairs: {effective_total_pairs}")
    print(f"Pre-existing cached pairs: {len(cached_old_pairs & required_pairs)}")
    print(f"Already computed pairs: {len(computed_pairs & required_pairs)}")
    print(f"Pairs queued for calculation: {num_tasks}")

    if num_tasks > 0:
        current_batch = []
        batch_id = next_batch_id
        
        pbar = tqdm(total=math.ceil(num_tasks / BATCH_SIZE), desc="Progress", unit="batch")
        for t in tasks_to_compute:
            current_batch.append(t)
            if len(current_batch) >= BATCH_SIZE:
                process_batch(current_batch, batch_id, WORKERS, NEW_EMBEDDINGS, has_old_paths, current_checksum)
                batch_id += 1; pbar.update(1)
                current_batch = []
                
        if current_batch:
            process_batch(current_batch, batch_id, WORKERS, NEW_EMBEDDINGS, has_old_paths, current_checksum)
            pbar.update(1)
        pbar.close()
        
    # Compile intermediate HDF5 batches into final output network
    compile_final_output(
        new_headers=new_headers,
        seq_lens=seq_lens,
        required_pairs=required_pairs,
        cached_old_pairs=cached_old_pairs,
        new_N=new_N,
        old_header_to_idx=old_header_to_idx,
        old_l_score=old_l_score,
        old_l_len=old_l_len,
        old_g_score=old_g_score,
        old_g_len=old_g_len,
        has_old_paths=has_old_paths,
        OLD_PATHS=OLD_PATHS,
        actual_idx_map=actual_idx_map
    )

if __name__ == "__main__":
    run_injection()