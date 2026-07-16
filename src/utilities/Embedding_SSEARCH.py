"""
File: Embedding_SSEARCH.py
===================================
Similar to the NCBI's traditional SSEARCH program, this tool performs a rigorous one-vs-all database search. However, instead of using a standard amino acid substitution matrix (like BLOSUM62), it aligns sequences based on the high-dimensional structural similarity of their Protein Language Model (pLM) embeddings.

This allows you to find remote homologs that share structural similarities even if their literal sequence identity has degraded entirely.

How it Works:
1. Target Database: You select a pre-computed embedding database (.h5) to search against. (If the .h5 file doesn't exist yet, the script will automatically generate one from the selected FASTA file).
2. Query Input: You can either type the exact header of a sequence already in the FASTA file, OR paste a brand new raw amino acid sequence into the 'Query Sequence' box.
3. Inference: The script calculates the structural embedding for your query.
4. Scanning: Using parallel CPU workers, it scans your query against every sequence in the database using either Local (Smith-Waterman) or Global (Needleman-Wunsch) dynamic programming.
5. Scoring: The raw alignment scores are normalized (to prevent bias toward excessively long sequences) and ranked.

Outputs:
The script generates two files in the same directory as your input FASTA:
- Report_<Query>.txt: A human-readable text file showing the ranked hits, their normalized scores, raw scores, and effective alignment lengths.
- Hits_<Query>.fasta: A clean FASTA file containing the sequences of all your top hits, ordered strictly by rank, with your query sequence pinned to the very top. This file is perfectly formatted to be immediately dropped into an MSA tool!

Key Parameters:
- Query Sequence (Optional): If you populate this box, the script will ignore the FASTA lookup and use your raw text.
- Norm Score Cutoff: Filters out any hits that fall below a specific normalized similarity score.
- Alignment Mode: Use 'local' if you are searching for a specific structural domain within larger proteins. Use 'global' if you are comparing overall holistic similarity.
"""
# %%
import os
import h5py
import numpy as np
import pandas as pd
import torch
import re
import sys
import Hardware_Utils
from numba import jit
from multiprocessing import Pool, set_start_method
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FASTA = "Sample.fasta"
INPUT_EMBED = "Sample_[E1_RA]_embeddings.h5"

# QUERY SETTINGS
QUERY_HEADER = "Query_Header"
QUERY_SEQUENCE = "" # Optional: If left blank, it fetches the sequence from INPUT_FASTA using QUERY_HEADER.
OUTPUT_NAME = "" # Optional: Custom base name for the generated output files.

# SEARCH PARAMETERS
TOP_K = 2500 
NORM_THRESHOLD = None
ALIGNMENT_MODE = "local"
LOCAL_GAP_P = -2.0
GLOBAL_GAP_P = 0.0
NORM_MODE = "longer_sequence"

# HARDWARE & CACHE
WORKERS = 8                  
SAVING_PRECISION = "float16" 

FASTA_DIR = os.path.join("..", "Input_Files", "Sequence_Sets")
EMBED_DIR = os.path.join("..", "Embeddings")
REPORT_DIR = os.path.join("..", "Cache_Files", "Align_Report")
GENERATE_FASTA = False

# --- JSON Settings Override ---
import json
import ast

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SETTINGS_FILE = os.path.join(PROJECT_ROOT, "Input_Files", "tools_settings.json")

if os.path.exists(SETTINGS_FILE):
    try:
        with open(SETTINGS_FILE, "r") as f:
            all_settings = json.load(f)
            
            if "DIRECTORIES" in all_settings:
                for k, v in all_settings["DIRECTORIES"].items():
                    if k in globals() and v is not None and str(v).strip() != "":
                        if not os.path.isabs(str(v)):
                            v = os.path.normpath(os.path.join(PROJECT_ROOT, str(v)))
                        globals()[k] = v
                        
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
FULL_INPUT_FASTA = os.path.join(FASTA_DIR, INPUT_FASTA) if FASTA_DIR else ""
FULL_INPUT_EMBED = os.path.join(EMBED_DIR, INPUT_EMBED) if EMBED_DIR else ""

_model_name = "unknown_model"
_match = re.search(r"_\[(.*?)\]_embeddings\.h5$", INPUT_EMBED)
if _match:
    _model_name = _match.group(1)
MODEL_NAME = _model_name

# --- 2. LIBRARY CHECK ---------------------------------------------------------
try:
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig
    from transformers import BertTokenizer, BertModel, T5Tokenizer, T5EncoderModel
except ImportError:
    print("\n[!] Warning: ESM or Transformers libraries not found.\n")

# --- 3. DATA & MODEL LOADING --------------------------------------------------
def read_fasta_to_dict(file_path):
    seq_dict = {}
    current_header, current_seq = None, []
    if not os.path.exists(file_path):
        print(f"Error: FASTA file not found at {file_path}")
        sys.exit(1)
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if current_header: seq_dict[current_header] = "".join(current_seq)
                current_header = line[1:]
                current_seq = []
            else: current_seq.append(line)
        if current_header: seq_dict[current_header] = "".join(current_seq)
    return seq_dict

def read_fasta_lists(file_path):
    headers, sequences = [], []
    d = read_fasta_to_dict(file_path)
    for h, s in d.items():
        headers.append(h); sequences.append(s)
    return headers, sequences

def load_model_integrated(model_name):
    device = Hardware_Utils.get_optimal_device()
    print(f"[System] Loading model '{model_name}' on {device}...")
    if "esmc" in model_name:
        return ESMC.from_pretrained(model_name).to(device), device, "esmc"
    elif "prot_bert" in model_name:
        tokenizer = BertTokenizer.from_pretrained(f"Rostlab/{model_name}", do_lower_case=False)
        model = BertModel.from_pretrained(f"Rostlab/{model_name}").to(device)
        model.eval()
        return (tokenizer, model), device, "bert"
    elif "ProstT5" in model_name:
        tokenizer = T5Tokenizer.from_pretrained(f"Rostlab/{model_name}_fp16", do_lower_case=False)
        model = T5EncoderModel.from_pretrained(f"Rostlab/{model_name}_fp16").to(device)
        return (tokenizer, model), device, "t5"
    else: raise ValueError(f"Unknown model: {model_name}")

def get_embedding_integrated(seq, model_obj, device, model_type, target_dtype):
    seq = seq.upper()
    
    # 1. Exact sanitization (same as Generate_Embeddings.py)
    match = re.search(r'[ACDEFGHIKLMNPQRSTVWYBZJXUO].*[ACDEFGHIKLMNPQRSTVWYBZJXUO]|[ACDEFGHIKLMNPQRSTVWYBZJXUO]', seq)
    seq = match.group(0) if match else ""
    
    if model_type == "esmc":
        seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY\-]', '-', seq)
    elif model_type in ["bert", "t5"]:
        seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWYBZJXUO\-]', 'X', seq)
        seq = re.sub(r'[BZUO]', 'X', seq) 

    # 2. Inference
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

def prepare_database_embeddings():
    if os.path.exists(FULL_INPUT_EMBED):
        print(f"[Init] Found cached HDF5 embeddings at:\n       {FULL_INPUT_EMBED}")
        with h5py.File(FULL_INPUT_EMBED, "r") as hf:
            raw_headers = hf['headers'][:]
            headers = [h.decode('utf-8') if isinstance(h, bytes) else h for h in raw_headers]
        return headers, None
        
    print(f"[Init] No HDF5 cache found. Generating '{SAVING_PRECISION}' embeddings from FASTA...")
    os.makedirs(os.path.dirname(FULL_INPUT_EMBED), exist_ok=True)
    headers, seqs = read_fasta_lists(FULL_INPUT_FASTA)
    model_obj, device, model_type = load_model_integrated(MODEL_NAME)
    
    target_dtype = np.float16 if SAVING_PRECISION == "float16" else np.float32
    
    # Write directly to HDF5 on-the-fly to save RAM
    with h5py.File(FULL_INPUT_EMBED, "w") as hf:
        hf.attrs["model_name"] = MODEL_NAME
        hf.attrs["num_sequences"] = len(headers)
        
        dt_str = h5py.string_dtype(encoding='utf-8')
        hf.create_dataset("headers", data=np.array(headers, dtype=object), dtype=dt_str)
        emb_group = hf.create_group("embeddings")
        
        for header, seq in tqdm(zip(headers, seqs), total=len(headers), desc="Embedding Generation"):
            emb = get_embedding_integrated(seq, model_obj, device, model_type, target_dtype)
            safe_h = header.replace("/", "_").replace("\\", "_")
            emb_group.create_dataset(safe_h, data=emb)
            
    return headers, (model_obj, device, model_type)

# --- 4. ALIGNMENT & NORMALIZATION LOGIC ---------------------------------------
def compute_score_matrix_torch(emb_i, emb_j, device):
    t_i = torch.as_tensor(emb_i, device=device, dtype=torch.float16)
    t_j = torch.as_tensor(emb_j, device=device, dtype=torch.float16)
    t_i_norm = torch.nn.functional.normalize(t_i, p=2, dim=-1)
    t_j_norm = torch.nn.functional.normalize(t_j, p=2, dim=-1)
    cos_sim = torch.mm(t_i_norm, t_j_norm.T)
    dist_mat = 1.0 - cos_sim
    sim_mat = torch.exp(-dist_mat)
    epsilon = 1e-8
    row_mean = sim_mat.mean(dim=1, keepdim=True); row_std = sim_mat.std(dim=1, keepdim=True)
    col_mean = sim_mat.mean(dim=0, keepdim=True); col_std = sim_mat.std(dim=0, keepdim=True)
    z_r = (sim_mat - row_mean) / (row_std + epsilon)
    z_c = (sim_mat - col_mean) / (col_std + epsilon)
    return ((z_r + z_c) / 2.0).to(dtype=torch.float32, device="cpu").numpy()

@jit(nopython=True, fastmath=True)
def run_traceback_numba(score_matrix, gap_p, is_local):
    N, M = score_matrix.shape
    dp = np.zeros((N + 1, M + 1), dtype=np.float32)
    pointer = np.zeros((N + 1, M + 1), dtype=np.int8) 

    if not is_local: 
        # GLOBAL (NW)
        for c in range(M + 1): dp[0, c] = c * gap_p; pointer[0, c] = 3
        for r in range(N + 1): dp[r, 0] = r * gap_p; pointer[r, 0] = 2
        pointer[0, 0] = 0
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                match = dp[i-1, j-1] + score_matrix[i-1, j-1]
                delete = dp[i-1, j] + gap_p
                insert = dp[i, j-1] + gap_p
                best = match; ptr = 1
                if delete > best: best = delete; ptr = 2
                if insert > best: best = insert; ptr = 3
                dp[i, j] = best; pointer[i, j] = ptr
        max_score = dp[N, M]; start_i, start_j = N, M
    else: 
        # LOCAL (SW)
        max_score_val = 0.0; start_i, start_j = 0, 0
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                match = dp[i-1, j-1] + score_matrix[i-1, j-1]
                delete = dp[i-1, j] + gap_p
                insert = dp[i, j-1] + gap_p
                best = 0.0; ptr = 0
                if match > best: best = match; ptr = 1
                if delete > best: best = delete; ptr = 2
                if insert > best: best = insert; ptr = 3
                dp[i, j] = best; pointer[i, j] = ptr
                if best > max_score_val: max_score_val = best; start_i, start_j = i, j
        max_score = max_score_val

    # Count path length for normalization
    i, j = start_i, start_j
    path_len = 0
    while True:
        if is_local and dp[i, j] == 0: break
        if not is_local and i == 0 and j == 0: break
        if i == 0 and j == 0: break
        p = pointer[i, j]
        path_len += 1
        if p == 1: i -= 1; j -= 1
        elif p == 2: i -= 1
        elif p == 3: j -= 1
        else: break
            
    return max_score, path_len

def normalize_score(raw_score, align_len, len_i, len_j, mode):
    if mode == "alignment_length": return raw_score / align_len if align_len > 0 else 0.0
    elif mode == "shorter_sequence": denom = min(len_i, len_j); return raw_score / denom if denom > 0 else 0.0
    elif mode == "longer_sequence": denom = max(len_i, len_j); return raw_score / denom if denom > 0 else 0.0
    elif mode == "average_sequence": denom = (len_i + len_j) / 2.0; return raw_score / denom if denom > 0 else 0.0
    else: return raw_score / align_len if align_len > 0 else 0.0

# --- HDF5 MULTIPROCESSING INITIALIZATION ---
worker_hf = None
def init_worker(h5_path):
    global worker_hf
    worker_hf = h5py.File(h5_path, "r", libver='latest', swmr=True)

def search_worker(args):
    idx, header, safe_h, q_emb, mode, gap, norm_mode = args
    global worker_hf
    
    device = Hardware_Utils.get_optimal_device()
    is_local = (mode == 'local')
    
    t_emb = worker_hf["embeddings"][safe_h][:]
    mat = compute_score_matrix_torch(q_emb, t_emb, device)
    
    if is_local: mat -= 2.0
    
    raw, path_len = run_traceback_numba(mat, gap, is_local)
    
    len_q = q_emb.shape[0]
    len_t = t_emb.shape[0]
    norm = normalize_score(raw, path_len, len_q, len_t, norm_mode)
    
    if norm_mode == "alignment_length": eff_len = path_len
    elif norm_mode == "shorter_sequence": eff_len = min(len_q, len_t)
    elif norm_mode == "longer_sequence": eff_len = max(len_q, len_t)
    elif norm_mode == "average_sequence": eff_len = (len_q + len_t) / 2.0
    else: eff_len = path_len

    return {"index": idx, "header": header, "raw_score": raw, "norm_score": norm, "length": eff_len, "seq_len": len_t, "aln_len": path_len}

# --- 5. REPORTING -------------------------------------------------------------
def save_results(df, query_meta, db_size, seq_lookup, base_filename, query_seq, norm_mode, gap_p):
    q_head, q_len = query_meta
    
    col_header = "ALN-LEN"

    # Base metadata block
    meta_lines = []
    meta_lines.append("="*80); meta_lines.append(f"{'PROTEIN EMBEDDING SEARCH REPORT':^80}"); meta_lines.append("="*80)
    meta_lines.append(f" Query:       {q_head}")
    meta_lines.append(f" Query Len:   {q_len} residues")
    meta_lines.append(f" Database:    {INPUT_EMBED} ({db_size} sequences)")
    meta_lines.append(f" Mode:        {ALIGNMENT_MODE.upper()} Alignment")
    meta_lines.append("-" * 80)
    meta_lines.append(f" Parameters:  Gap Penalty = {gap_p}")
    meta_lines.append(f" Metric:      Raw Score / {norm_mode}")
    meta_lines.append(f" Filters:     Top_K={TOP_K} | Norm_Threshold={NORM_THRESHOLD}")
    meta_lines.append("-" * 80 + "\n")
    
    report_lines = list(meta_lines)
    onscreen_lines = list(meta_lines)
    
    if df.empty:
        report_lines.append("  [No hits found satisfying the criteria]")
        onscreen_lines.append("  [No hits found satisfying the criteria]")
        xlsx_data = []
    else:
        table_hdr = [
            f" {'RANK':<6} | {'NORM-SCR':<9} | {'RAW':<9} | {'SEQ-LEN':<8} | {col_header:<8} | {'HEADER'}",
            f"{'-'*7}-+-{'-'*9}-+-{'-'*9}-+-{'-'*8}-+-{'-'*8}-+-{'-'*35}"
        ]
        report_lines.extend(table_hdr)
        onscreen_lines.extend(table_hdr)
        
        printed_hits = 0
        rank_counter = 1
        xlsx_data = []
        for i, row in df.iterrows():
            if row['index'] == -1: continue 
            head = row['header']
            seq_len_val = int(row['seq_len'])
            aln_len_val = int(row['aln_len'])
            norm_score = row['norm_score']
            raw_score = row['raw_score']
            
            row_line = f" {rank_counter:<6} | {norm_score:<9.3f} | {raw_score:<9.1f} | {seq_len_val:<8} | {aln_len_val:<8} | {head}"
            report_lines.append(row_line)
            if printed_hits < 100:
                onscreen_lines.append(row_line)
                printed_hits += 1
                
            xlsx_data.append({
                "Rank": rank_counter,
                "Norm Score": float(norm_score),
                "Raw Score": float(raw_score),
                "Seq Len": int(seq_len_val),
                "Aln Len": int(aln_len_val),
                "Header": str(head)
            })
            
            rank_counter += 1
            
        if len(df) - 1 > 100:
            onscreen_lines.append(f"\n  [Onscreen display limited to first 100 hits. The full list of {len(df) - 1} hits is stored in the report file.]")
            
    print("\n".join(onscreen_lines))
    
    output_dir = REPORT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    report_text = "\n".join(report_lines)
    with open(os.path.join(output_dir, f"Report_{base_filename}.txt"), "w") as f: f.write(report_text)
    
    # Generate and save Excel Report
    xlsx_path = os.path.join(output_dir, f"Report_{base_filename}.xlsx")
    if not xlsx_data:
        xlsx_df = pd.DataFrame(columns=["Rank", "Norm Score", "Raw Score", "Seq Len", "Aln Len", "Header"])
    else:
        xlsx_df = pd.DataFrame(xlsx_data)
        
    meta_data = [
        {"Parameter": "Query Header", "Value": q_head},
        {"Parameter": "Query Length (residues)", "Value": q_len},
        {"Parameter": "Database", "Value": INPUT_EMBED},
        {"Parameter": "Database Size (sequences)", "Value": db_size},
        {"Parameter": "Alignment Mode", "Value": ALIGNMENT_MODE},
        {"Parameter": "Gap Penalty", "Value": gap_p},
        {"Parameter": "Normalization Mode", "Value": norm_mode},
        {"Parameter": "Norm Score Cutoff", "Value": NORM_THRESHOLD if NORM_THRESHOLD is not None else "None"},
        {"Parameter": "Top K", "Value": TOP_K if TOP_K is not None else "None"}
    ]
    meta_df = pd.DataFrame(meta_data)
    
    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            xlsx_df.to_excel(writer, sheet_name="Search Results", index=False)
            meta_df.to_excel(writer, sheet_name="Search Parameters", index=False)
        print(f"[Export] Excel report saved to: {xlsx_path}")
    except Exception as e:
        print(f"[Warning] Failed to save Excel report: {e}")
    
    if GENERATE_FASTA:
        # OUTPUT FASTA (Ranked with query at the top)
        count = 1 
        with open(os.path.join(output_dir, f"Hits_{base_filename}.fasta"), "w") as f:
            f.write(f">{q_head}\n{query_seq}\n")
            if not df.empty and seq_lookup:
                for i, row in df.iterrows():
                    if row['index'] == -1: continue
                    head = row['header']
                    if head in seq_lookup:
                        f.write(f">{head}\n{seq_lookup[head]}\n")
                        count += 1
        print(f"[Export] {count} sequences exported to Hits_{base_filename}.fasta (Query is #1)")

# --- 6. MAIN ------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[Init] Loading sequences for export...")
    seq_lookup = read_fasta_to_dict(FULL_INPUT_FASTA)
    
    db_headers, loaded_model_pack = prepare_database_embeddings()
    
    # Process Query Input
    query_name = QUERY_HEADER if QUERY_HEADER else "Manual_Query"
    query_seq_str = QUERY_SEQUENCE.strip() if QUERY_SEQUENCE else ""
    query_emb = None
    target_dtype = np.float16 if SAVING_PRECISION == "float16" else np.float32

    if not query_seq_str:
        if QUERY_HEADER in seq_lookup:
            query_seq_str = seq_lookup[QUERY_HEADER]
            print(f"[Input] Found query sequence for '{QUERY_HEADER}' in FASTA.")
            
            # ---> NEW: Check the HDF5 Database for pre-calculated embedding <---
            safe_q_head = QUERY_HEADER.replace("/", "_").replace("\\", "_")
            if os.path.exists(FULL_INPUT_EMBED):
                with h5py.File(FULL_INPUT_EMBED, "r") as hf:
                    if "embeddings" in hf and safe_q_head in hf["embeddings"]:
                        print(f"[Input] Found pre-calculated embedding for '{QUERY_HEADER}' in HDF5 cache. Skipping generation.")
                        query_emb = hf["embeddings"][safe_q_head][:].astype(target_dtype)
        else:
            print(f"[Error] QUERY_SEQUENCE is empty and QUERY_HEADER '{QUERY_HEADER}' not found in FASTA.")
            sys.exit(1)
    else:
        print(f"[Input] Using manually provided query sequence.")
        
    # ---> CONDITIONAL EMBEDDING GENERATION <---
    # Only run the model inference if we didn't successfully pull the embedding from the HDF5 file
    if query_emb is None:
        print(f"[Input] Generating new embedding for the query...")
        if loaded_model_pack is None: model_obj, device, model_type = load_model_integrated(MODEL_NAME)
        else: model_obj, device, model_type = loaded_model_pack
        
        query_emb = get_embedding_integrated(query_seq_str, model_obj, device, model_type, target_dtype)

    # 3. Search
    gap_p = LOCAL_GAP_P if ALIGNMENT_MODE == "local" else GLOBAL_GAP_P
    tasks = []
    
    for i, header in enumerate(db_headers):
        safe_h = header.replace("/", "_").replace("\\", "_")
        tasks.append((i, header, safe_h, query_emb, ALIGNMENT_MODE, gap_p, NORM_MODE))
        
    print(f"[Search] Scanning {len(tasks)} sequences against {MODEL_NAME} using '{NORM_MODE}' normalization...")
    results = []
    
    try: set_start_method('spawn')
    except RuntimeError: pass
    
    with Pool(processes=WORKERS, initializer=init_worker, initargs=(FULL_INPUT_EMBED,)) as pool:
        for res in tqdm(pool.imap_unordered(search_worker, tasks, chunksize=50), total=len(tasks)):
            results.append(res)
            
    df = pd.DataFrame(results)
    df = df.sort_values(by="norm_score", ascending=False)
    
    # 4. Filter
    if NORM_THRESHOLD is not None: 
        df = df[df['norm_score'] >= float(NORM_THRESHOLD)]
        
    # Check if the query already exists in the database
    query_in_db = query_name in db_headers
    
    # Remove the query from the database hits to prevent duplicates in the FASTA
    df = df[df['header'] != query_name]
    
    if TOP_K is not None:
        # If in DB, we want TOP_K total sequences in the FASTA (1 pinned query + TOP_K-1 hits)
        # If not in DB, we want TOP_K+1 total sequences (1 pinned query + TOP_K hits)
        limit = int(TOP_K) - 1 if query_in_db else int(TOP_K)
        limit = max(0, limit)
        
        if len(df) > limit:
            df = df.head(limit)
    
    # Add dummy row for Query (Ensures it appears at the top of the text report)
    q_row = pd.DataFrame([{"index": -1, "header": f"(Query) {query_name}", "raw_score": 0.0, "norm_score": 99.9, "length": len(query_emb), "seq_len": len(query_emb), "aln_len": len(query_emb)}])
    df = pd.concat([q_row, df], ignore_index=True)
    
    # Use custom name if provided, otherwise fallback to sanitized query name
    if OUTPUT_NAME and str(OUTPUT_NAME).strip():
        base_filename = str(OUTPUT_NAME).strip()
    else:
        base_filename = re.sub(r'[^a-zA-Z0-9]', '_', query_name)[:20]
        
    save_results(df, (query_name, len(query_emb)), len(db_headers), seq_lookup, base_filename, query_seq_str, NORM_MODE, gap_p)