# %% Import Necessary Libraries
import os
import sys
import numpy as np
import h5py
import torch
from tqdm import tqdm
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

# Import E1 Model Components
try:
    from E1.batch_preparer import E1BatchPreparer
    from E1.modeling import E1ForMaskedLM
    HAS_E1 = True
except ImportError:
    HAS_E1 = False

# ==========================================
# CONFIGURATION
# ==========================================
OLD_SEQUENCE_SET = "Expansin_GH5_2300"
NEW_SEQUENCE_SET = "Expansin_GH5_2300"
NETWORK_MODEL_NAME = "esmc_600m"

# --- Embedding Generation Parameters ---
BATCH_SIZE = 1                         
E1_MODEL_NAME = "Profluent-Bio/E1-600m"

# --- Paths & Mode Discovery ---
import glob
# Find OLD_HDF5 dynamically by searching for sequence set and E1 modes in the filename
_old_pattern = os.path.join("Embeddings", f"{OLD_SEQUENCE_SET}_[E1_*]_embeddings.h5")
_found_files = glob.glob(_old_pattern)
OLD_HDF5 = _found_files[0] if _found_files else os.path.join("Embeddings", f"{OLD_SEQUENCE_SET}_[E1_RA]_embeddings.h5")

INPUT_FASTA = os.path.join("Sequence_Sets", f"{NEW_SEQUENCE_SET}.fasta")
INPUT_NETWORK_H5 = os.path.join("Networks", f"{NEW_SEQUENCE_SET}_[{NETWORK_MODEL_NAME}]_network.h5")

# Determine embedding mode from OLD_HDF5 metadata (model_name) or fallback
embedding_mode = "RA" # default fallback
if os.path.exists(OLD_HDF5):
    try:
        with h5py.File(OLD_HDF5, "r") as _hf_meta:
            if "model_name" in _hf_meta.attrs:
                model_name_attr = _hf_meta.attrs["model_name"]
                E1_MODEL_NAME = model_name_attr
                if "RA" in model_name_attr:
                    embedding_mode = "RA"
                elif "NORM" in model_name_attr:
                    embedding_mode = "NORM"
                else:
                    # Fallback if model name doesn't contain RA or NORM
                    embedding_mode = "NORM" if "_[E1_NORM]_" in OLD_HDF5 else "RA"
            else:
                # If model_name is missing, fallback to filename check
                embedding_mode = "NORM" if "_[E1_NORM]_" in OLD_HDF5 else "RA"
    except Exception:
        pass
else:
    # If the file does not exist, check the filename to see if NORM is in it
    if "_[E1_NORM]_" in OLD_HDF5:
        embedding_mode = "NORM"

OUTPUT_HDF5 = os.path.join("Embeddings", f"{NEW_SEQUENCE_SET}_[E1_{embedding_mode}]_embeddings.h5")

# --- Homolog Search Parameters (For RA Mode) ---
MAX_SEQ_LENGTH = 4096                  
TOP_K_SEARCH = 1000                    
SELECTION_STRIDE = 20                  
SCORE_MODE = "global"                  
NORMALIZATION_MODE = "alignment_length"
MAX_HOMOLOGS = 50   # Exact number of homologs to append

# --- Sparse Imputation Parameters (For RA Mode with Sparse Network) ---
POOLING_METHOD = "max"                  # "mean" or "max"
LENGTH_RATIO_POWER = 2.0                # Exponent for length penalty
SIMILARITY_MODEL_NAME = None            # Embedding model name to use for cosine similarity (e.g. "esmc_600m"). If None, defaults to NETWORK_MODEL_NAME.

# --- Embedding-Only Homolog Selection Switch ---
USE_ONLY_EMBEDDINGS = False             # If True, ignore network file and select homologs using only embedding similarity

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def read_fasta(file_path):
    if not os.path.exists(file_path): sys.exit(f"❌ FASTA file not found: {file_path}")
    headers_list, seq_dict = [], {}
    current_header, current_sequence = None, []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if current_header:
                    seq_dict[current_header] = "".join(current_sequence)
                    headers_list.append(current_header)
                current_header = line[1:]
                current_sequence = []
            else:
                current_sequence.append(line)
        if current_header:
            seq_dict[current_header] = "".join(current_sequence)
            headers_list.append(current_header)
    return headers_list, seq_dict

def extract_homologs_for_targets(new_target_headers, net_headers, hf_net):
    print(f"\n--- Extracting Context for {len(new_target_headers)} New Sequences ({DEVICE}) ---")
    
    if USE_ONLY_EMBEDDINGS:
        print("Using ONLY embedding similarity for homolog selection (ignoring network file).")
        model_name_for_sim = SIMILARITY_MODEL_NAME if SIMILARITY_MODEL_NAME is not None else NETWORK_MODEL_NAME
        INPUT_EMBED_H5 = os.path.join(os.path.dirname(OUTPUT_HDF5), f"{NEW_SEQUENCE_SET}_[{model_name_for_sim}]_embeddings.h5")
        if not os.path.exists(INPUT_EMBED_H5):
            sys.exit(f"❌ Error: Embedding-only mode active, but base embedding file not found at: {INPUT_EMBED_H5}\n"
                     f"Please make sure you have generated the embeddings for {model_name_for_sim} first.")
            
        print(f"Loading base embeddings from {INPUT_EMBED_H5}...")
        mean_embs = []
        actual_seq_lens = []
        with h5py.File(INPUT_EMBED_H5, "r") as f_emb:
            raw_headers = f_emb['headers'][:]
            headers = [h.decode('utf-8') if isinstance(h, bytes) else h for h in raw_headers]
            
            has_emb_group = "embeddings" in f_emb
            filtered_headers = []
            for h in tqdm(headers, desc="Loading Embeddings"):
                safe_h = h.replace("/", "_").replace("\\", "_")
                if has_emb_group:
                    emb = f_emb["embeddings"][safe_h][:]
                else:
                    emb = f_emb[safe_h][:]
                
                seq_len = emb.shape[0]
                if seq_len <= MAX_SEQ_LENGTH:
                    actual_seq_lens.append(seq_len)
                    filtered_headers.append(h)
                    if POOLING_METHOD == "max":
                        pooled = np.max(emb, axis=0)
                    else:
                        pooled = np.mean(emb, axis=0)
                    mean_embs.append(pooled)
                    
        num_valid = len(filtered_headers)
        print(f"Loaded {num_valid} valid sequences (<= {MAX_SEQ_LENGTH} AA) from embeddings.")
        if num_valid == 0:
            sys.exit("❌ Error: No valid sequences remaining after length filtering.")
            
        mean_embs = np.array(mean_embs, dtype=np.float32)
        actual_seq_lens = np.array(actual_seq_lens, dtype=np.float32)
        
        print("Calculating all-vs-all normalized embedding cosine similarities...")
        norms = np.linalg.norm(mean_embs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        norm_embs = mean_embs / norms
        cos_sim_mat = np.dot(norm_embs, norm_embs.T)
        cos_sim_mat = np.clip(cos_sim_mat, -1.0, 1.0)
        
        # Apply sequence length ratio adjustment if configured
        if LENGTH_RATIO_POWER != 0.0:
            lens_col = actual_seq_lens[:, np.newaxis]
            lens_row = actual_seq_lens[np.newaxis, :]
            min_lens = np.minimum(lens_col, lens_row)
            max_lens = np.maximum(lens_col, lens_row)
            max_lens = np.maximum(max_lens, 1)
            length_ratio_mat = min_lens / max_lens
            if LENGTH_RATIO_POWER != 1.0:
                length_ratio_mat = length_ratio_mat ** LENGTH_RATIO_POWER
            cos_sim_mat = cos_sim_mat * length_ratio_mat
            
        similarity_matrix = torch.tensor(cos_sim_mat, device=DEVICE, dtype=torch.float32)
        similarity_matrix.fill_diagonal_(float('-inf'))
        
    else:
        # Original network-based logic
        # ---> NEW: Detect if the input is a BLAST E-Value network <---
        is_blast = "EValue" in os.path.basename(INPUT_NETWORK_H5) or "Evalue" in os.path.basename(INPUT_NETWORK_H5) or "blast" in os.path.basename(INPUT_NETWORK_H5).lower()
        if is_blast:
            print("Detected BLAST/E-Value Network. Bypassing embedding normalization.")

        if 'seq_lens' in hf_net:
            seq_lens = hf_net['seq_lens'][:]
        else:
            seq_lens = np.zeros(len(net_headers), dtype=np.int32)

        num_original = len(net_headers)
        src_indices = hf_net['i'][:].astype(np.int64)
        dst_indices = hf_net['j'][:].astype(np.int64)
        
        # ---> NEW: Safely extract scores based on network type <---
        if is_blast:
            raw_scores = hf_net['score'][:].astype(np.float32)
            raw_lens = None
        else:
            if SCORE_MODE == "global":
                raw_scores = hf_net['g_score'][:].astype(np.float32)
                raw_lens   = hf_net['g_len'][:].astype(np.float32)
            else:
                raw_scores = hf_net['l_score'][:].astype(np.float32)
                raw_lens   = hf_net['l_len'][:].astype(np.float32)

        # Filter Sequences > MAX_SEQ_LENGTH
        valid_mask = seq_lens <= MAX_SEQ_LENGTH
        valid_indices = np.where(valid_mask)[0]
        num_valid = len(valid_indices)
        
        if num_valid == 0:
            sys.exit("❌ Error: No valid sequences remaining after length filtering.")

        old_to_new_map = np.full(num_original, -1, dtype=np.int64)
        old_to_new_map[valid_indices] = np.arange(num_valid)

        # Build Dense Similarity Matrix for the whole network
        print("Constructing Master Similarity Matrix...")
        
        valid_edge_mask = (valid_mask[src_indices]) & (valid_mask[dst_indices])
        final_src = old_to_new_map[src_indices[valid_edge_mask]]
        final_dst = old_to_new_map[dst_indices[valid_edge_mask]]
        final_scores = torch.tensor(raw_scores[valid_edge_mask], device=DEVICE, dtype=torch.float32)

        t_src = torch.tensor(final_src, device=DEVICE, dtype=torch.long)
        t_dst = torch.tensor(final_dst, device=DEVICE, dtype=torch.long)
        
        # ---> NEW: Bypass Normalization entirely for BLAST E-Values <---
        if is_blast:
            normalized_scores = final_scores
        else:
            if NORMALIZATION_MODE == "alignment_length":
                align_lens = torch.tensor(raw_lens[valid_edge_mask], device=DEVICE, dtype=torch.float32)
                denom = align_lens + 1e-6
            else:
                valid_lens_gpu = torch.tensor(seq_lens[valid_indices], device=DEVICE, dtype=torch.float32)
                len_src, len_dst = valid_lens_gpu[t_src], valid_lens_gpu[t_dst]
                
                if NORMALIZATION_MODE == "longer_sequence": denom = torch.maximum(len_src, len_dst) + 1e-6
                elif NORMALIZATION_MODE == "shorter_sequence": denom = torch.minimum(len_src, len_dst) + 1e-6
                elif NORMALIZATION_MODE == "average_sequence": denom = ((len_src + len_dst) / 2.0) + 1e-6
                else: sys.exit(f"❌ Error: Unknown normalization mode '{NORMALIZATION_MODE}'")

            normalized_scores = final_scores / denom

        is_sparse = len(final_src) < int(num_valid * (num_valid - 1) / 2)
        
        if is_sparse and not is_blast:
            print(f"Network is sparse ({len(final_src)} / {int(num_valid * (num_valid - 1) / 2)} edges). Activating hybrid cosine-alignment transformation...")
            model_name_for_sim = SIMILARITY_MODEL_NAME if SIMILARITY_MODEL_NAME is not None else NETWORK_MODEL_NAME
            INPUT_EMBED_H5 = os.path.join(os.path.dirname(OUTPUT_HDF5), f"{NEW_SEQUENCE_SET}_[{model_name_for_sim}]_embeddings.h5")
            if not os.path.exists(INPUT_EMBED_H5):
                sys.exit(f"❌ Error: Sparse network detected, but base embedding file not found at: {INPUT_EMBED_H5}\n"
                         f"Please make sure you have generated the embeddings for {model_name_for_sim} first.")
                
            print(f"Loading base embeddings from {INPUT_EMBED_H5}...")
            mean_embs = []
            actual_seq_lens = []
            filtered_headers = [net_headers[i] for i in valid_indices]
            with h5py.File(INPUT_EMBED_H5, "r") as f_emb:
                has_emb_group = "embeddings" in f_emb
                for h in tqdm(filtered_headers, desc="Computing Pooled Embeddings"):
                    safe_h = h.replace("/", "_").replace("\\", "_")
                    if has_emb_group:
                        emb = f_emb["embeddings"][safe_h][:]
                    else:
                        emb = f_emb[safe_h][:]
                    
                    actual_seq_lens.append(emb.shape[0])
                    if POOLING_METHOD == "max":
                        pooled = np.max(emb, axis=0)
                    else:
                        pooled = np.mean(emb, axis=0)
                    mean_embs.append(pooled)
                    
            mean_embs = np.array(mean_embs, dtype=np.float32)
            actual_seq_lens = np.array(actual_seq_lens, dtype=np.float32)
            
            print("Calculating all-vs-all length-adjusted similarities (length ratio * cosine similarity)...")
            norms = np.linalg.norm(mean_embs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            norm_embs = mean_embs / norms
            cos_sim_mat = np.dot(norm_embs, norm_embs.T)
            cos_sim_mat = np.clip(cos_sim_mat, -1.0, 1.0)
            
            # Apply sequence length ratio adjustment
            lens_col = actual_seq_lens[:, np.newaxis]
            lens_row = actual_seq_lens[np.newaxis, :]
            min_lens = np.minimum(lens_col, lens_row)
            max_lens = np.maximum(lens_col, lens_row)
            max_lens = np.maximum(max_lens, 1)
            length_ratio_mat = min_lens / max_lens
            
            if LENGTH_RATIO_POWER != 1.0:
                length_ratio_mat = length_ratio_mat ** LENGTH_RATIO_POWER
                
            cos_sim_mat = cos_sim_mat * length_ratio_mat
            
            # Extract overlapping pairs
            X_cos = cos_sim_mat[final_src, final_dst]
            Y_align = normalized_scores.cpu().numpy()
            
            print("Fitting Isotonic Regression (Adjusted Similarity -> Alignment Score)...")
            if len(X_cos) > 100000:
                np.random.seed(42)
                sample_idx = np.random.choice(len(X_cos), size=100000, replace=False)
                X_fit = X_cos[sample_idx]
                Y_fit = Y_align[sample_idx]
            else:
                X_fit = X_cos
                Y_fit = Y_align
                
            iso_reg = IsotonicRegression(out_of_bounds='clip')
            iso_reg.fit(X_fit, Y_fit)
            
            # Evaluate fitness on all edges
            Y_pred = iso_reg.predict(X_cos)
            rho, _ = spearmanr(X_cos, Y_align)
            r2 = r2_score(Y_align, Y_pred)
            print(f"Isotonic Regression Fit Diagnostics:")
            print(f"  > Spearman Rank Correlation (rho): {rho:.4f}")
            print(f"  > Coefficient of Determination (R^2): {r2:.4f}")
            
            print("Imputing missing similarity scores for all-vs-all pairs...")
            dense_scores = iso_reg.predict(cos_sim_mat.ravel()).reshape(num_valid, num_valid)
            dense_scores = 0.5 * (dense_scores + dense_scores.T)
            
            similarity_matrix = torch.tensor(dense_scores, device=DEVICE, dtype=torch.float32)
        else:
            similarity_matrix = torch.full((num_valid, num_valid), float('-inf'), device=DEVICE, dtype=torch.float32)
            
        similarity_matrix[t_src, t_dst] = normalized_scores
        similarity_matrix[t_dst, t_src] = normalized_scores 
        similarity_matrix.fill_diagonal_(float('-inf'))
        
        del src_indices, dst_indices, raw_scores, raw_lens, final_src, final_dst, final_scores, t_src, t_dst
        torch.cuda.empty_cache()
        filtered_headers = [net_headers[i] for i in valid_indices]

    # Map target headers to their valid matrix row indices
    print("Selecting Homologs for New Targets...")
    net_header_to_idx = {h: i for i, h in enumerate(net_headers)}
    
    target_matrix_indices = []
    active_targets = []
    
    for h in new_target_headers:
        if h in net_header_to_idx:
            if USE_ONLY_EMBEDDINGS:
                mat_idx = net_header_to_idx[h]
            else:
                orig_idx = net_header_to_idx[h]
                mat_idx = old_to_new_map[orig_idx]
            if mat_idx != -1:
                target_matrix_indices.append(mat_idx)
                active_targets.append(h)
                
    if not target_matrix_indices:
        return {}

    target_matrix_indices = torch.tensor(target_matrix_indices, device=DEVICE, dtype=torch.long)
    target_similarity = similarity_matrix[target_matrix_indices]

    actual_k = min(TOP_K_SEARCH, num_valid - 1) 
    target_strides = torch.arange(SELECTION_STRIDE - 1, TOP_K_SEARCH, SELECTION_STRIDE, device=DEVICE)
    if actual_k < SELECTION_STRIDE:
        target_strides = torch.tensor([0], device=DEVICE); actual_k = 1

    _, top_indices = torch.topk(target_similarity, k=actual_k, dim=1)
    
    valid_target_mask = target_strides < actual_k
    valid_target_indices = target_strides[valid_target_mask]
    selected_homologs = top_indices[:, valid_target_indices].cpu().numpy()
    
    homolog_map = {}
    for i, h in enumerate(active_targets):
        homolog_map[h] = [filtered_headers[idx] for idx in selected_homologs[i]]
        
    return homolog_map

def load_e1_model(model_name, device):
    print(f"\nLoading {model_name} on {device}...")
    model = E1ForMaskedLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()
    batch_preparer = E1BatchPreparer()
    return model, batch_preparer

def get_e1_embeddings_batch(input_strings, model, batch_preparer, device, target_dtype):
    batch = batch_preparer.get_batch_kwargs(input_strings, device=device)
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            outputs = model(
                input_ids=batch["input_ids"],
                within_seq_position_ids=batch["within_seq_position_ids"],
                global_position_ids=batch["global_position_ids"],
                sequence_ids=batch["sequence_ids"],
                use_cache=False, output_attentions=False, output_hidden_states=False,
            )

    embeddings = outputs.embeddings 
    last_seq_ids = batch["sequence_ids"].max(dim=1)[0][:, None]
    last_sequence_mask = batch["sequence_ids"] == last_seq_ids
    valid_residue_mask = ~batch_preparer.get_boundary_token_mask(batch["input_ids"])
    query_mask = last_sequence_mask & valid_residue_mask

    batch_results = []
    for i in range(len(input_strings)):
        emb_i = embeddings[i, query_mask[i]]
        batch_results.append(emb_i.cpu().float().numpy().astype(target_dtype))

    return batch_results

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    if not HAS_E1:
        print("❌ Error: Could not import 'E1'. Make sure the E1 repository is installed via 'pip install -e .'")
        sys.exit(1)
        
    print(f"--- 🧬 E1 Embedding Injection [{embedding_mode} Mode] ---")
    
    # 1. Load Original Metadata
    if not os.path.exists(OLD_HDF5): sys.exit(f"❌ Error: Old embedding not found: {OLD_HDF5}")
    with h5py.File(OLD_HDF5, "r") as hf_old:
        old_headers = [h.decode('utf-8') if isinstance(h, bytes) else h for h in hf_old['headers'][:]]
        target_dtype = hf_old["embeddings"][old_headers[0]].dtype
    old_set = set(old_headers)
    print(f"Loaded {len(old_headers)} original embeddings. (Precision: {target_dtype})")

    # 2. Load New FASTA
    new_fasta_headers, seq_dict = read_fasta(INPUT_FASTA)
    fasta_set = set(new_fasta_headers)
    print(f"Loaded {len(new_fasta_headers)} target sequences from FASTA.")

    # 3. Load New Network Headers
    if USE_ONLY_EMBEDDINGS:
        model_name_for_sim = SIMILARITY_MODEL_NAME if SIMILARITY_MODEL_NAME is not None else NETWORK_MODEL_NAME
        INPUT_EMBED_H5 = os.path.join(os.path.dirname(OUTPUT_HDF5), f"{NEW_SEQUENCE_SET}_[{model_name_for_sim}]_embeddings.h5")
        if not os.path.exists(INPUT_EMBED_H5):
            sys.exit(f"❌ Error: Embedding-only mode active, but base embedding file not found at: {INPUT_EMBED_H5}")
        with h5py.File(INPUT_EMBED_H5, "r") as f_emb:
            raw_headers = f_emb['headers'][:]
            net_headers = [h.decode('utf-8') if isinstance(h, bytes) else h for h in raw_headers]
        hf_net = None
        net_set = set(net_headers)
    else:
        if not os.path.exists(INPUT_NETWORK_H5): sys.exit(f"❌ Error: Network not found: {INPUT_NETWORK_H5}")
        hf_net = h5py.File(INPUT_NETWORK_H5, "r")
        net_headers = [h.decode('utf-8') if isinstance(h, bytes) else h for h in hf_net['headers'][:]]
        net_set = set(net_headers)
    
    # --- STRICT VALIDATION CHECKS ---
    print("\n--- Validating Inputs ---")
    if fasta_set != net_set:
        sys.exit(f"❌ CRITICAL ERROR: New FASTA and New Network sequences DO NOT MATCH perfectly.\n"
                 f"   FASTA Count: {len(fasta_set)} | Network Count: {len(net_set)}")
        
    if not old_set.issubset(fasta_set):
        missing = old_set - fasta_set
        sys.exit(f"❌ CRITICAL ERROR: {len(missing)} sequences from the original embedding "
                 f"are missing from the new datasets. Example: '{list(missing)[0]}'")
    print("✅ Validation Passed: Old embeddings are a perfect subset of the new synchronized inputs.")

    # 4. Identify New Sequences
    new_targets = list(fasta_set - old_set)
    print(f"\nIdentified {len(new_targets)} NEW sequences requiring E1 calculation.")

    # 5. Extract Homologs (Only for the NEW sequences)
    homolog_map = {}
    if len(new_targets) > 0 and embedding_mode == "RA":
        homolog_map = extract_homologs_for_targets(new_targets, net_headers, hf_net)
        
    # We can close the network file now
    if hf_net is not None:
        hf_net.close()

    # 6. Load Model (Only if necessary)
    model, preparer = None, None
    if len(new_targets) > 0:
        model, preparer = load_e1_model(E1_MODEL_NAME, DEVICE)

    # 7. Stream & Inject
    os.makedirs(os.path.dirname(OUTPUT_HDF5), exist_ok=True)
    print(f"\nWriting injected embeddings to {OUTPUT_HDF5}...")
    
    successful_headers = []
    
    with h5py.File(OLD_HDF5, "r") as hf_in, h5py.File(OUTPUT_HDF5, "w") as hf_out:
        if E1_MODEL_NAME.startswith("E1_"):
            hf_out.attrs["model_name"] = E1_MODEL_NAME
        else:
            hf_out.attrs["model_name"] = f"E1_{embedding_mode}"
        emb_group_in = hf_in["embeddings"]
        emb_group_out = hf_out.create_group("embeddings")
        
        # We will iterate through the NEW FASTA to guarantee the final output order
        batch_input_strs = []
        valid_batch_headers = []
        
        for h in tqdm(new_fasta_headers, desc="Processing Sequences"):
            
            # --- COPY OLD ---
            if h in old_set:
                emb_group_out.create_dataset(h, data=emb_group_in[h][:])
                successful_headers.append(h)
                continue
                
            # --- PREPARE NEW ---
            query_seq = seq_dict[h]
            homolog_seqs = []
            
            if embedding_mode == "RA" and h in homolog_map:
                h_headers = homolog_map[h][:MAX_HOMOLOGS]
                for h_header in h_headers:
                    if h_header in seq_dict:
                        homolog_seqs.append(seq_dict[h_header])
            
            combined_list = homolog_seqs + [query_seq]
            batch_input_strs.append(",".join(combined_list))
            valid_batch_headers.append(h)
            
            # --- EXECUTE BATCH ---
            if len(batch_input_strs) >= BATCH_SIZE:
                try:
                    batch_embs = get_e1_embeddings_batch(batch_input_strs, model, preparer, DEVICE, target_dtype)
                    for head, emb in zip(valid_batch_headers, batch_embs):
                        emb_group_out.create_dataset(head, data=emb)
                        successful_headers.append(head)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"\n⚠️ OOM on Batch. Recovering sequentially...")
                        torch.cuda.empty_cache()
                        for single_str, head in zip(batch_input_strs, valid_batch_headers):
                            try:
                                single_emb = get_e1_embeddings_batch([single_str], model, preparer, DEVICE, target_dtype)[0]
                                emb_group_out.create_dataset(head, data=single_emb)
                                successful_headers.append(head)
                            except RuntimeError:
                                print(f"\n❌ Final OOM on sequence {head}. Skipping.")
                                torch.cuda.empty_cache()
                                continue
                    else:
                        print(f"\n❌ Error on batch: {e}")
                
                # Reset Buffer
                batch_input_strs = []
                valid_batch_headers = []
                
        # --- PROCESS REMAINING BUFFER ---
        if len(batch_input_strs) > 0:
            try:
                batch_embs = get_e1_embeddings_batch(batch_input_strs, model, preparer, DEVICE, target_dtype)
                for head, emb in zip(valid_batch_headers, batch_embs):
                    emb_group_out.create_dataset(head, data=emb)
                    successful_headers.append(head)
            except RuntimeError as e:
                print(f"\n❌ Final Buffer Error: {e}")

        # Save Final Metadata
        dt_str = h5py.string_dtype(encoding='utf-8')
        hf_out.create_dataset("headers", data=np.array(successful_headers, dtype=object), dtype=dt_str)
        hf_out.attrs["num_sequences"] = len(successful_headers)
        
    print(f"\n✅ Complete! {len(successful_headers)} embeddings successfully injected and written to {OUTPUT_HDF5}.")