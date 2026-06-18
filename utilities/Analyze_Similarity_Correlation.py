import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score

# Configuration of input files
EMBED_PATH = r"Embeddings\Foldtype_IV_ATAs_13000_[E1_RA]_embeddings.h5"
NETWORK_PATH = r"Input_Files\Networks_EValues\Foldtype_IV_ATAs_13000_[E1_RA]_network.h5"
OUTPUT_DIR = r"Results\Similarity_Correlation_Analysis"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# Ensure output directories exist
os.makedirs(PLOTS_DIR, exist_ok=True)

def get_edge_cos_sim(embs, i_indices, j_indices):
    """Memory-efficient edge-wise cosine similarity calculation."""
    vecs_i = embs[i_indices]
    vecs_j = embs[j_indices]
    norms_i = np.linalg.norm(vecs_i, axis=1)
    norms_j = np.linalg.norm(vecs_j, axis=1)
    dot_prod = np.sum(vecs_i * vecs_j, axis=1)
    
    # Avoid division by zero
    denominator = np.maximum(norms_i * norms_j, 1e-8)
    sims = dot_prod / denominator
    return np.clip(sims, -1.0, 1.0)

def main():
    print("1. Loading HDF5 network metadata...")
    if not os.path.exists(NETWORK_PATH):
        raise FileNotFoundError(f"Network file not found at: {NETWORK_PATH}")
    if not os.path.exists(EMBED_PATH):
        raise FileNotFoundError(f"Embeddings file not found at: {EMBED_PATH}")
        
    with h5py.File(NETWORK_PATH, "r") as f_net:
        raw_net_headers = f_net['headers'][:]
        net_headers = [h.decode('utf-8') if isinstance(h, bytes) else h for h in raw_net_headers]
        arr_i = f_net['i'][:]
        arr_j = f_net['j'][:]
        g_scores = f_net['g_score'][:]
        g_lens = f_net['g_len'][:]
        l_scores = f_net['l_score'][:]
        l_lens = f_net['l_len'][:]

    num_total_edges = len(arr_i)
    print(f"  > Total edges in network file: {num_total_edges}")

    # Subsample 500,000 edges randomly to represent all embeddings
    print("Subsampling 500,000 network edges for regression analysis...")
    np.random.seed(42)
    E_sub = min(500000, num_total_edges)
    sample_indices = np.random.choice(num_total_edges, size=E_sub, replace=False)

    sub_i = arr_i[sample_indices].astype(np.int32)
    sub_j = arr_j[sample_indices].astype(np.int32)
    sub_g_score = g_scores[sample_indices]
    sub_g_len = g_lens[sample_indices]
    sub_l_score = l_scores[sample_indices]
    sub_l_len = l_lens[sample_indices]

    # Free up RAM
    del arr_i, arr_j, g_scores, g_lens, l_scores, l_lens

    print("2. Pre-calculating pooled vectors and sequence lengths for all embeddings...")
    with h5py.File(EMBED_PATH, "r") as f_emb:
        raw_emb_headers = f_emb['headers'][:]
        emb_headers = [h.decode('utf-8') if isinstance(h, bytes) else h for h in raw_emb_headers]
        
        # Intersect embedding and network headers
        common_set = set(net_headers).intersection(set(emb_headers))
        print(f"  > Common headers size: {len(common_set)} / {len(emb_headers)}")
        
        valid_headers = [h for h in net_headers if h in common_set]
        header_to_idx = {h: i for i, h in enumerate(valid_headers)}
        old_to_valid = {i: header_to_idx[h] for i, h in enumerate(net_headers) if h in header_to_idx}

        # Inspect dimensions dynamically
        safe_first = valid_headers[0].replace("/", "_").replace("\\", "_")
        emb_dim = f_emb["embeddings"][safe_first].shape[1]
        print(f"  > Embedding dimension: {emb_dim}")

        seq_lens = np.zeros(len(valid_headers), dtype=np.int32)
        mean_embs = np.zeros((len(valid_headers), emb_dim), dtype=np.float32)
        max_embs = np.zeros((len(valid_headers), emb_dim), dtype=np.float32)

        for idx, h in enumerate(tqdm(valid_headers, desc="Pooling Embeddings")):
            safe_h = h.replace("/", "_").replace("\\", "_")
            emb = f_emb["embeddings"][safe_h][:] # shape: (length, dim)
            seq_lens[idx] = emb.shape[0]
            mean_embs[idx] = np.mean(emb, axis=0)
            max_embs[idx] = np.max(emb, axis=0)

    # Map the subsampled edges to the valid intersecting subset
    map_arr = np.full(len(net_headers), -1, dtype=np.int32)
    for old_idx, val_idx in old_to_valid.items():
        map_arr[old_idx] = val_idx

    valid_edge_mask = (map_arr[sub_i] != -1) & (map_arr[sub_j] != -1)
    sub_i_valid = map_arr[sub_i[valid_edge_mask]]
    sub_j_valid = map_arr[sub_j[valid_edge_mask]]
    
    sub_g_score = sub_g_score[valid_edge_mask]
    sub_g_len = sub_g_len[valid_edge_mask]
    sub_l_score = sub_l_score[valid_edge_mask]
    sub_l_len = sub_l_len[valid_edge_mask]
    
    print(f"  > Valid subsampled edges after intersection mapping: {len(sub_i_valid)}")

    # Extract edge-wise lengths
    L_i = seq_lens[sub_i_valid]
    L_j = seq_lens[sub_j_valid]
    min_L = np.minimum(L_i, L_j)
    max_L = np.maximum(L_i, L_j)
    mean_L = (L_i + L_j) / 2.0

    print("Precomputing edge cosine similarities...")
    cos_mean = get_edge_cos_sim(mean_embs, sub_i_valid, sub_j_valid)
    cos_max = get_edge_cos_sim(max_embs, sub_i_valid, sub_j_valid)

    # Sampling 100,000 edges for isotonic regression fit speed
    sample_size = min(len(sub_i_valid), 100000)
    np.random.seed(42)
    fit_sample_idx = np.random.choice(len(sub_i_valid), size=sample_size, replace=False)

    results_list = []
    
    score_types = ["global", "local"]
    pooling_methods = ["mean", "max"]
    align_norm_methods = ["alignment_length", "shorter_sequence", "longer_sequence", "average_sequence"]
    embed_penalty_methods = ["none", "linear_ratio", "squared_ratio", "exp_max", "exp_mean"]

    total_runs = len(score_types) * len(pooling_methods) * len(align_norm_methods) * len(embed_penalty_methods)
    print(f"\n3. Executing Grid Search ({total_runs} combinations)...")

    run_idx = 0
    for score_type in score_types:
        base_scores = sub_g_score if score_type == "global" else sub_l_score
        base_lens = sub_g_len if score_type == "global" else sub_l_len
        
        for pooling in pooling_methods:
            base_sims = cos_mean if pooling == "mean" else cos_max
            
            for align_norm in align_norm_methods:
                # Normalization of Alignment scores
                if align_norm == "alignment_length":
                    Y = base_scores / np.maximum(base_lens, 1e-6)
                elif align_norm == "shorter_sequence":
                    Y = base_scores / np.maximum(min_L, 1e-6)
                elif align_norm == "longer_sequence":
                    Y = base_scores / np.maximum(max_L, 1e-6)
                elif align_norm == "average_sequence":
                    Y = base_scores / np.maximum(mean_L, 1e-6)
                    
                for embed_penalty in embed_penalty_methods:
                    # Normalization of Cosine Similarities
                    if embed_penalty == "none":
                        X = base_sims
                    elif embed_penalty == "linear_ratio":
                        X = base_sims * (min_L / np.maximum(max_L, 1e-6))
                    elif embed_penalty == "squared_ratio":
                        X = base_sims * ((min_L / np.maximum(max_L, 1e-6)) ** 2)
                    elif embed_penalty == "exp_max":
                        X = base_sims * np.exp(-np.abs(L_i - L_j) / np.maximum(max_L, 1e-6))
                    elif embed_penalty == "exp_mean":
                        X = base_sims * np.exp(-np.abs(L_i - L_j) / np.maximum(mean_L, 1e-6))
                        
                    # Isotonic regression fitting
                    iso = IsotonicRegression(out_of_bounds='clip')
                    iso.fit(X[fit_sample_idx], Y[fit_sample_idx])
                    Y_pred = iso.predict(X)
                    
                    # Spearman rho & R^2
                    rho, _ = spearmanr(X, Y)
                    r2 = r2_score(Y, Y_pred)
                    
                    results_list.append({
                        "score_type": score_type,
                        "pooling": pooling,
                        "align_norm": align_norm,
                        "embed_penalty": embed_penalty,
                        "spearman_rho": rho,
                        "r2_score": r2
                    })
                    
                    # Generate plot
                    plt.figure(figsize=(10, 6))
                    # Sample 5000 points to avoid plotting lag
                    plot_idx = np.random.choice(len(X), size=min(len(X), 5000), replace=False)
                    plt.scatter(X[plot_idx], Y[plot_idx], color='blue', alpha=0.3, label='Edges (Sample)', s=5)
                    
                    # Generate isotonic regression line
                    x_line = np.linspace(np.min(X), np.max(X), 1000)
                    y_line = iso.predict(x_line)
                    plt.plot(x_line, y_line, color='red', linewidth=3, label='Isotonic Fit')
                    
                    title_str = (
                        f"Isotonic Fit: {score_type.upper()} | {pooling.upper()} | {align_norm} | {embed_penalty}\n"
                        f"(rho = {rho:.4f}, R^2 = {r2:.4f})"
                    )
                    plt.title(title_str)
                    plt.xlabel("Length-Normalized Embedding Similarity")
                    plt.ylabel("Length-Normalized Alignment Score")
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.5)
                    plt.tight_layout()
                    
                    filename = f"{score_type}_{pooling}_{align_norm}_{embed_penalty}.png"
                    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
                    plt.close()
                    
                    run_idx += 1
                    if run_idx % 10 == 0:
                        print(f"  > Completed {run_idx}/{total_runs} combinations...")

    # Save summary report as CSV
    df_results = pd.DataFrame(results_list)
    df_results = df_results.sort_values(by="spearman_rho", ascending=False)
    df_results.to_csv(os.path.join(OUTPUT_DIR, "summary_report.csv"), index=False)
    
    print("\n=======================================================")
    print("TOP 10 COMBINATIONS BY SPEARMAN CORRELATION (rho):")
    print("=======================================================")
    print(df_results.head(10).to_string(index=False))
    print("=======================================================")
    print(f"\nAll plots saved to: {PLOTS_DIR}")
    print(f"Summary report saved to: {os.path.join(OUTPUT_DIR, 'summary_report.csv')}")

if __name__ == "__main__":
    main()
