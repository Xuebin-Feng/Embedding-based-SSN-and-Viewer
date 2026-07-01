import Command_Engine
import numpy as np
import matplotlib.pyplot as plt
import SSN_Utils as utils

def print_help():
    print("""
    Topology Clustering Tool
    ========================
    Usage: cluster [MODE] [PARAM_1] [MIN_SIZE]
           cluster list
           cluster help

    Modes:
      leiden (Default)
          - Leiden Community Detection (Modularity & Density optimization).
          - Automatically uses network edge scores as structural weights if available.
          - Requires: pip install leidenalg igraph
          - PARAM_1: Resolution (Higher = more clusters). Default: 1.0
          - Example: cluster leiden 1.0 10  (OR simply: cluster 1.0 10)
          
      mcl
          - Markov Clustering Algorithm (Simulates random walks).
          - Automatically uses network edge scores as structural weights if available.
          - Requires: pip install markov_clustering networkx
          - PARAM_1: Inflation (1.1 - 10.0, higher = tighter clusters). Default: 2.0
          - Example: cluster mcl 2.0 10

      jaccard
          - Filters edges based on shared neighbors and cuts weak connections.
          - Requires: Numba (JIT compilation)
          - PARAM_1: Threshold (0.0 - 1.0). Default: 0.2
          - Example: cluster jaccard 0.3 10
          
    Commands:
      list          - Prints current cluster statistics and node distributions to the console.

    Arguments:
      [MIN_SIZE]    (Optional) Minimum Cluster Size (Integer).
                    - Groups smaller than this are designated as 'Noise' (Cluster -1).
                    - Default: 10
    """)

def run(viewer, args):
    # --- 1. Help Check ---
    if args and args[0].lower() in ['help', '-h', '--help']:
        print_help()
        if hasattr(viewer, 'console_text'):
            viewer.console_text.text = "Help information printed to the terminal"
        return

    # --- LIST COMMAND ---
    if args and args[0].lower() == 'list':
        if getattr(viewer, 'cluster_labels', None) is None:
            msg = "No clusters are currently defined."
            Command_Engine.print_help(viewer, msg)
            return
            
        labels = viewer.cluster_labels
        n_nodes = viewer.n_nodes
        print(f"\n--- Current Cluster Statistics (Total Nodes: {n_nodes}) ---")
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_counts = dict(zip(unique_labels, counts))
        
        noise_count = label_counts.get(-1, 0)
        print(f"Noise (Unclustered): {noise_count} nodes ({noise_count/n_nodes*100:.2f}%)")
        
        print(f"\n{'='*52}")
        print(f"--- Current Cluster Statistics (Total: {n_nodes}) ---")
        print(f"{'='*52}")
        print(f"| {'Cluster Name':<20} | {'Node Count':>10} | {'Percent':>10} |")
        print(f"|{'-'*22}+{'-'*12}+{'-'*12}|")
        
        noise_count = label_counts.get(-1, 0)
        noise_pct = (noise_count / n_nodes) * 100
        print(f"| {'Noise (Unclustered)':<20} | {noise_count:>10} | {noise_pct:>9.2f}% |")
        
        sorted_clusters = sorted([k for k in label_counts.keys() if k != -1])
        for cid in sorted_clusters:
            c_count = label_counts[cid]
            c_pct = (c_count / n_nodes) * 100
            print(f"| {f'Cluster {cid}':<20} | {c_count:>10} | {c_pct:>9.2f}% |")
        print(f"{'='*52}\n")
        
        msg = f"Listed {len(sorted_clusters)} clusters in console."
        viewer.console_text.text = msg
        return

    # --- 2. Parse Arguments ---
    mode = "leiden"
    param1 = None
    min_sz = 10
    
    if len(args) >= 1:
        first_arg = args[0].lower()
        if first_arg in ['jaccard', 'mcl', 'leiden']:
            mode = first_arg
            if len(args) >= 2: 
                try: param1 = float(args[1])
                except ValueError: print("Error: Parameter must be a number."); return
            if len(args) >= 3: 
                try: min_sz = int(args[2])
                except ValueError: print("Error: Min Size must be an integer."); return
        else:
            # Fallback to default Jaccard logic if first argument is a number
            try: param1 = float(args[0])
            except ValueError:
                print(f"Error: Unknown mode or invalid number '{args[0]}'")
                return
            if len(args) >= 2: 
                try: min_sz = int(args[1])
                except ValueError: print("Error: Min Size must be an integer."); return

    # Apply defaults if param1 wasn't provided
    if param1 is None:
        if mode == "jaccard": param1 = 0.2
        elif mode == "mcl": param1 = 2.0
        elif mode == "leiden": param1 = 1.0

    n_nodes = viewer.n_nodes
    edges = np.array(viewer.edges, dtype=np.int32)
    labels = np.full(n_nodes, -1, dtype=int)
    
    viewer.console_text.text = f"Clustering ({mode.upper()})..."
    print(f"Running {mode.upper()} Clustering (Param={param1}, MinSize={min_sz})...")

    # =======================================================
    # MODE 1: JACCARD (Topology Filtering + BFS)
    # =======================================================
    if mode == "jaccard":
        thresh = param1
        if not utils.NUMBA_AVAILABLE:
            print("Error: Numba required for topology clustering.")
            viewer.console_text.text = "Error: Numba library missing."
            return

        # Prepare adjacency data for Numba
        degrees = np.zeros(n_nodes, dtype=np.int32)
        for u, v in edges:
            degrees[u] += 1; degrees[v] += 1
            
        indptr = np.zeros(n_nodes + 1, dtype=np.int32)
        indptr[1:] = np.cumsum(degrees)
        indices = np.zeros(indptr[-1], dtype=np.int32)
        
        temp_counts = np.zeros(n_nodes, dtype=np.int32)
        for u, v in edges:
            indices[indptr[u] + temp_counts[u]] = v; temp_counts[u] += 1
            indices[indptr[v] + temp_counts[v]] = u; temp_counts[v] += 1
            
        for i in range(n_nodes): 
            indices[indptr[i]:indptr[i+1]].sort()

        # Numba Filter
        keep_mask = utils.fast_jaccard_filter(edges, indptr, indices, thresh)
        
        filtered_adj = {i: [] for i in range(n_nodes)}
        for i, keep in enumerate(keep_mask):
            if keep:
                u, v = edges[i]
                filtered_adj[u].append(v)
                filtered_adj[v].append(u)

        # BFS Connected Components
        visited = np.zeros(n_nodes, dtype=bool)
        cluster_id = 0
        
        for i in range(n_nodes):
            if not visited[i]:
                stack = [i]
                visited[i] = True
                component = []
                
                while stack:
                    node = stack.pop()
                    component.append(node)
                    for neighbor in filtered_adj[node]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            stack.append(neighbor)
                
                if len(component) >= min_sz:
                    for node in component: 
                        labels[node] = cluster_id
                    cluster_id += 1

    # =======================================================
    # MODE 2: MARKOV CLUSTERING (MCL)
    # =======================================================
    elif mode == "mcl":
        inflation = param1
        try:
            import markov_clustering as mc
            import networkx as nx
            import scipy.sparse as sp
        except ImportError:
            msg = "Missing libraries! Run: pip install markov_clustering networkx scipy"
            print(f"Error: {msg}")
            viewer.console_text.text = msg
            return
            
        print("Building Sparse Adjacency Matrix...")
        row = np.concatenate([edges[:, 0], edges[:, 1]])
        col = np.concatenate([edges[:, 1], edges[:, 0]])
        
        # Optionally use edge scores if they exist in the viewer, else default to 1
        if hasattr(viewer, 'edge_scores'):
            d_vals = np.concatenate([viewer.edge_scores, viewer.edge_scores])
        else:
            d_vals = np.ones(len(row))
            
        matrix = sp.csr_matrix((d_vals, (row, col)), shape=(n_nodes, n_nodes))
        
        # ---> NEW: Safely suppress SciPy sparsity warnings triggered by MCL <---
        import warnings
        from scipy.sparse import SparseEfficiencyWarning
        warnings.simplefilter("ignore", category=SparseEfficiencyWarning)
        
        print(f"Running MCL (Inflation = {inflation}). This may take a moment...")
        result = mc.run_mcl(matrix, inflation=inflation)
        clusters = mc.get_clusters(result)
        
        cluster_id = 0
        for comp in clusters:
            if len(comp) >= min_sz:
                for node in comp:
                    labels[node] = cluster_id
                cluster_id += 1

    # =======================================================
    # MODE 3: LEIDEN COMMUNITY DETECTION
    # =======================================================
    elif mode == "leiden":
        resolution = param1
        try:
            import igraph as ig
            import leidenalg as la
        except ImportError:
            msg = "Missing libraries! Run: pip install leidenalg igraph"
            print(f"Error: {msg}")
            viewer.console_text.text = msg
            return
            
        print("Building Graph & Mapping Edge Weights...")
        G = ig.Graph(n=n_nodes, edges=edges.tolist())
        
        # Automatically harness the physics edge weights if available!
        if hasattr(viewer, 'edge_scores'):
            G.es['weight'] = viewer.edge_scores
            print(f"Running Leiden (Resolution = {resolution}, Weighted).")
            partition = la.find_partition(G, la.RBConfigurationVertexPartition, resolution_parameter=resolution, weights='weight')
        else:
            print(f"Running Leiden (Resolution = {resolution}, Unweighted).")
            partition = la.find_partition(G, la.RBConfigurationVertexPartition, resolution_parameter=resolution)
            
        cluster_id = 0
        for comp in partition:
            if len(comp) >= min_sz:
                for node in comp:
                    labels[node] = cluster_id
                cluster_id += 1

    # --- 6. Update Viewer ---
    
    # ---> NEW: Save the state before applying new clusters and colors for Undo/Redo
    viewer._save_state()
    
    viewer.cluster_labels = labels
    
    # Store parameters as strings so external commands (align.py, etc.) know what was used
    viewer.last_cluster_params = (f"{mode.upper()}_{param1}", min_sz)
    
    # Apply Colors
    cmap = plt.get_cmap('tab20')
    for i in range(n_nodes):
        lbl = labels[i]
        if lbl == -1: 
            viewer.current_colors[i] = (0.8, 0.8, 0.8, 0.4) # Grey for noise
        else: 
            viewer.current_colors[i] = cmap(lbl % 20)
        
    viewer.update_nodes()
    
    # --- 7. Print Statistics ---
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    
    print(f"\n{'='*52}")
    print(f"--- {mode.upper()} Clustering Stats (Total Nodes: {n_nodes}) ---")
    print(f"{'='*52}")
    print(f"| {'Cluster Name':<20} | {'Node Count':>10} | {'Percent':>10} |")
    print(f"|{'-'*22}+{'-'*12}+{'-'*12}|")
    
    noise_count = label_counts.get(-1, 0)
    noise_pct = (noise_count / n_nodes) * 100
    print(f"| {'Noise (Unclustered)':<20} | {noise_count:>10} | {noise_pct:>9.2f}% |")
    
    sorted_clusters = sorted([k for k in label_counts.keys() if k != -1])
    for cid in sorted_clusters:
        c_count = label_counts[cid]
        c_pct = (c_count / n_nodes) * 100
        print(f"| {f'Cluster {cid}':<20} | {c_count:>10} | {c_pct:>9.2f}% |")
    print(f"{'='*52}\n")
    
    n_clusters = len(sorted_clusters)
    msg = f"Done! Found {n_clusters} clusters via {mode.upper()}."
    viewer.console_text.text = msg
    print(msg)