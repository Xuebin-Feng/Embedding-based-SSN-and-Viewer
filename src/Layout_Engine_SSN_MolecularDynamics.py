import numpy as np
import math
from collections import deque

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    import torch
    try:
        from utilities import Hardware_Utils
    except ImportError:
        import Hardware_Utils
    HAS_TORCH = True
except Exception as e:
    import traceback
    print("Warning: PyTorch or Hardware_Utils could not be imported. GPU acceleration will be disabled.")
    print(f"Detail: {e}")
    traceback.print_exc()
    HAS_TORCH = False

# --- 1. Physics Kernels ---

def _get_physics_kernel():
    def _run_physics_kernel(pos, vel, springs, comp_labels, box_limit, dt, damping, k_spr, k_coul, max_f, cutoff_dist):
        n_balls = pos.shape[0]
        acc = np.zeros_like(pos)
        
        # Calculate squared cutoff for efficient distance comparison
        cutoff_sq = cutoff_dist * cutoff_dist 
        
        # --- SPRINGS (Attraction) ---
        for i in range(springs.shape[0]):
            idx_a, idx_b = springs[i, 0], springs[i, 1]
            dx, dy = pos[idx_a, 0] - pos[idx_b, 0], pos[idx_a, 1] - pos[idx_b, 1]
            dist = math.sqrt(dx*dx + dy*dy) + 1e-9
            
            f = -k_spr * dist
            
            acc[idx_a, 0] += f * (dx/dist); acc[idx_a, 1] += f * (dy/dist)
            acc[idx_b, 0] -= f * (dx/dist); acc[idx_b, 1] -= f * (dy/dist)
            
        # --- REPULSION (Coulomb Only) ---
        for i in range(n_balls):
            for j in range(i+1, n_balls):
                dx, dy = pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]
                dist_sq = dx*dx + dy*dy
                
                if dist_sq > cutoff_sq: continue 
                if dist_sq == 0.0: continue 

                dist = math.sqrt(dist_sq)
                safe_dist = max(dist, 0.5) 
                
                f = k_coul / (safe_dist**2)
                
                if f > max_f: f = max_f
                
                acc[i, 0] += f*(dx/dist); acc[i, 1] += f*(dy/dist)
                acc[j, 0] -= f*(dx/dist); acc[j, 1] -= f*(dy/dist)
                    
        # --- INTEGRATION (Euler) ---
        rmsd = 0.0
        for i in range(n_balls):
            acc[i] -= damping * vel[i]
            vel[i] += acc[i] * dt
            old_p = pos[i].copy()
            pos[i] += vel[i] * dt
            
            if pos[i,0] > box_limit: pos[i,0]=box_limit; vel[i,0]*=-0.5
            elif pos[i,0] < -box_limit: pos[i,0]=-box_limit; vel[i,0]*=-0.5
            if pos[i,1] > box_limit: pos[i,1]=box_limit; vel[i,1]*=-0.5
            elif pos[i,1] < -box_limit: pos[i,1]=-box_limit; vel[i,1]*=-0.5
            
            diff = pos[i] - old_p
            rmsd += diff[0]**2 + diff[1]**2
            
        return math.sqrt(rmsd / n_balls)
        
    if NUMBA_AVAILABLE:
        return jit(nopython=True, fastmath=True)(_run_physics_kernel)
    return _run_physics_kernel

run_physics_kernel = _get_physics_kernel()

class SSNSimulationCPU:
    def __init__(self, pos, springs, comp_labels, box_limit, params):
        self.pos = pos.astype(np.float32)
        self.vel = np.zeros_like(pos)
        self.springs = springs
        self.comp_labels = comp_labels
        self.box = box_limit
        self.params = params
        
    def step(self, current_step, apply_warmup=True):
        max_cutoff = self.params.get('COULOMB_CUTOFF', 15.0)
        max_steps = self.params.get('MAX_STEPS', 2000)
        
        if apply_warmup:
            target_step = max_steps / 4.0
            curve_scale = target_step / 5.0 
            
            if current_step >= target_step:
                cutoff = max_cutoff
            else:
                num = math.atan(current_step / curve_scale)
                den = math.atan(target_step / curve_scale)
                cutoff = max_cutoff * (num / den)
        else:
            cutoff = max_cutoff

        return run_physics_kernel(
            self.pos, self.vel, self.springs, self.comp_labels, self.box, 
            self.params.get('DT', 0.1), 
            self.params.get('DAMPING', 0.5), 
            self.params.get('SPRING_K', 0.1), 
            self.params.get('COULOMB_K', 50.0), 
            self.params.get('MAX_FORCE_LIMIT', 10.0), 
            cutoff 
        )
        
    def get_pos(self): return self.pos

if HAS_TORCH:
    class SSNSimulationGPU:
        def __init__(self, pos, springs, comp_labels, box_limit, params):
            self.device = Hardware_Utils.get_optimal_device()
            self.pos = torch.tensor(pos, dtype=torch.float32, device=self.device)
            self.vel = torch.zeros_like(self.pos)
            self.springs = torch.tensor(springs, dtype=torch.long, device=self.device)
            self.comp_labels = torch.tensor(comp_labels, dtype=torch.long, device=self.device)
            self.box = box_limit
            self.params = params
        
        @torch.no_grad()
        def step(self, current_step, apply_warmup=True):
            max_cutoff = self.params.get('COULOMB_CUTOFF', 15.0)
            max_steps = self.params.get('MAX_STEPS', 2000)
            
            if apply_warmup:
                target_step = max_steps / 4.0
                curve_scale = target_step / 5.0 
                
                if current_step >= target_step:
                    cutoff = max_cutoff
                else:
                    num = math.atan(current_step / curve_scale)
                    den = math.atan(target_step / curve_scale)
                    cutoff = max_cutoff * (num / den)
            else:
                cutoff = max_cutoff

            # --- PHYSICS ---
            delta = self.pos.unsqueeze(1) - self.pos.unsqueeze(0)
            dist = delta.norm(dim=2) + 1e-9
            
            # Element-wise force magnitude calculation
            f_mag = self.params.get('COULOMB_K', 50.0) / (dist.clamp(min=0.5)**2)
            
            # Zero out forces outside the cutoff or self-repulsion using torch.where
            is_self = torch.eye(dist.size(0), dtype=torch.bool, device=self.device)
            cond = (dist < cutoff) & (~is_self)
            f_mag = torch.where(cond, f_mag, 0.0)
            
            f_mag = f_mag.clamp(max=self.params.get('MAX_FORCE_LIMIT', 10.0))
            acc = (f_mag.unsqueeze(2) * (delta/dist.unsqueeze(2))).sum(dim=1)
            
            if len(self.springs) > 0:
                idx_a, idx_b = self.springs[:,0], self.springs[:,1]
                pa, pb = self.pos[idx_a], self.pos[idx_b]
                d = (pa-pb).norm(dim=1) + 1e-9
                f = -self.params.get('SPRING_K', 0.1) * d
                fv = f.unsqueeze(1) * ((pa-pb)/d.unsqueeze(1))
                acc.index_add_(0, idx_a, fv); acc.index_add_(0, idx_b, -fv)
            
            damping = self.params.get('DAMPING', 0.5)
            dt = self.params.get('DT', 0.1)
            
            acc -= damping * self.vel
            self.vel += acc * dt
            old = self.pos.clone()
            self.pos += self.vel * dt
            
            # --- Boundary Collisions (Match CPU Bouncing) ---
            out_of_bounds_x = self.pos[:, 0].abs() > self.box
            out_of_bounds_y = self.pos[:, 1].abs() > self.box
            
            # Reverse and dampen velocity for nodes hitting the walls
            self.vel[out_of_bounds_x, 0] *= -0.5
            self.vel[out_of_bounds_y, 1] *= -0.5
            
            # Clamp positions
            self.pos.clamp_(min=-self.box, max=self.box)
            
            rmsd = (self.pos - old).norm(dim=1).pow(2).mean().sqrt().item()
            
            del delta, dist, cond, is_self, f_mag, acc, old
            return rmsd

        def get_pos(self): return self.pos.cpu().numpy()

# --- 2. Components & Packing Logic ---

def find_connected_components(n_nodes, edges):
    """Finds all independent subgraphs using Breadth-First Search."""
    adj = {i: [] for i in range(n_nodes)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    visited = np.zeros(n_nodes, dtype=bool)
    components = []
    
    for i in range(n_nodes):
        if not visited[i]:
            comp = []
            q = [i]
            visited[i] = True
            while q:
                curr = q.pop(0)
                comp.append(curr)
                for neighbor in adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        q.append(neighbor)
            components.append(comp)
    return components

def get_component_labels(n_nodes, edges):
    """Maps each node to its connected component ID for isolated physics."""
    components = find_connected_components(n_nodes, edges)
    labels = np.zeros(n_nodes, dtype=np.int32)
    for c_id, comp in enumerate(components):
        for node in comp:
            labels[node] = c_id
    return labels

def pack_components_to_grid(pos, edges, n_nodes, grid_size, padding, packing_geometry="Square"):
    """Packs independent network components into a strict master grid layout."""
    print("Packing independent components using macro-grid boolean packing...")
    components = find_connected_components(n_nodes, edges)
    
    if not components:
        return pos, 100.0

    # --- 1. Map edges to components for fast lookup ---
    node_to_comp = {}
    for c_id, comp in enumerate(components):
        for node in comp:
            node_to_comp[node] = c_id
            
    comp_edges = {c_id: [] for c_id in range(len(components))}
    for u, v in edges:
        c_id = node_to_comp.get(u)
        if c_id is not None:
            comp_edges[c_id].append((u, v))

    comp_info = []
    for c_id, comp in enumerate(components):
        idx = np.array(comp)
        comp_pos = pos[idx]
        
        # Use a top-left origin approach so Y goes downwards into grid rows
        min_x = np.min(comp_pos[:, 0])
        max_y = np.max(comp_pos[:, 1]) 
        
        shifted_pos = comp_pos - [min_x, max_y] # X is >= 0, Y is <= 0
        
        global_to_local = {g: l for l, g in enumerate(comp)}
        points = list(shifted_pos)
        
        # Rasterize edges so we don't accidentally place a dot on a connecting line
        for u, v in comp_edges[c_id]:
            p1 = shifted_pos[global_to_local[u]]
            p2 = shifted_pos[global_to_local[v]]
            dist = np.hypot(p2[0]-p1[0], p2[1]-p1[1])
            
            # Sample points along the edge line
            steps = int(dist / (grid_size / 4)) + 1
            for i in range(1, steps):
                t = i / steps
                px = p1[0] + t * (p2[0] - p1[0])
                py = p1[1] + t * (p2[1] - p1[1])
                points.append([px, py])
                
        # --- Determine Grid Footprint ---
        max_c, max_r = 0, 0
        pad = padding / 2.0
        
        # First pass: find the maximum grid cells required
        for px, py in points:
            pos_y = -py # Invert Y so positive goes down into rows
            c_max = int((px + pad) / grid_size)
            r_max = int((pos_y + pad) / grid_size)
            max_c = max(max_c, c_max)
            max_r = max(max_r, r_max)
            
        cols = max_c + 1
        rows = max_r + 1
        mask = np.zeros((rows, cols), dtype=bool)
        
        # Second pass: mark cells as occupied
        for px, py in points:
            pos_y = -py
            c_min = int(max(0, px - pad) / grid_size)
            c_max = int((px + pad) / grid_size)
            r_min = int(max(0, pos_y - pad) / grid_size)
            r_max = int((pos_y + pad) / grid_size)
            mask[r_min:r_max+1, c_min:c_max+1] = True
            
        comp_info.append({
            'indices': idx,
            'shifted_pos': shifted_pos,
            'mask': mask,
            'cols': cols,
            'rows': rows,
            'area': np.sum(mask),
            'num_nodes': len(idx)
        })
        
    # --- 2. Sort components: Largest area first, then tie-break by node count ---
    comp_info.sort(key=lambda x: (x['area'], x['num_nodes']), reverse=True)
    
    # --- 3. Prepare the Master Global Grid and Run Spiral Placement ---
    total_area = sum(c['area'] for c in comp_info)
    max_cols = max(c['cols'] for c in comp_info)
    max_rows = max(c['rows'] for c in comp_info)
    
    is_circle = (packing_geometry.lower() == "circle")
    multiplier = 2.0 if is_circle else 1.5
    
    # Start with a grid size S estimated from total area, scaled to prevent border clipping
    S = max(int(math.ceil(math.sqrt(total_area) * multiplier)), max_cols, max_rows)
    
    new_pos = np.zeros((n_nodes, 2), dtype=np.float32)
    # Fill unconnected nodes first
    new_pos[:] = pos[:]
    
    # Center nodes aesthetically within their grid squares
    center_x_offset = grid_size / 2.0  
    center_y_offset = -grid_size / 2.0
    
    while True:
        grid_map = np.zeros((S, S), dtype=bool)
        center_r = S // 2
        center_c = S // 2
        
        # Calculate physical center coordinate
        if is_circle:
            center_x_phys = center_c + 0.5 * (center_r % 2)
            center_y_phys = -center_r * (math.sqrt(3.0) / 2.0)
        else:
            center_x_phys = center_c
            center_y_phys = -center_r
            
        # Generate all coordinates in the grid map
        coords = []
        for r in range(S):
            for c in range(S):
                if is_circle:
                    x_phys = c + 0.5 * (r % 2)
                    y_phys = -r * (math.sqrt(3.0) / 2.0)
                    dist = (x_phys - center_x_phys)**2 + (y_phys - center_y_phys)**2
                else:
                    dist_l_inf = max(abs(r - center_r), abs(c - center_c))
                    dist_l_2 = (r - center_r)**2 + (c - center_c)**2
                    dist = (dist_l_inf, dist_l_2)
                coords.append((r, c, dist))
                
        # Sort coords by distance from center (ascending)
        coords.sort(key=lambda x: x[2])
        
        success = True
        placed_offsets = []
        
        for comp in comp_info:
            mask = comp['mask']
            h, w = comp['rows'], comp['cols']
            placed = False
            
            for r_center, c_center, _ in coords:
                # Target top-left row/col so that component center aligns close to r_center, c_center
                r = r_center - h // 2
                c = c_center - w // 2
                
                if r >= 0 and r + h <= S and c >= 0 and c + w <= S:
                    if not np.any(grid_map[r:r+h, c:c+w] & mask):
                        grid_map[r:r+h, c:c+w] |= mask
                        placed_offsets.append((r, c))
                        placed = True
                        break
                        
            if not placed:
                success = False
                break
                
        if success:
            # Apply offsets
            for comp, (r, c) in zip(comp_info, placed_offsets):
                if is_circle:
                    # Hexagonal physical coordinates
                    x_offset = (c + 0.5 * (r % 2)) * grid_size
                    y_offset = -r * grid_size * (math.sqrt(3.0) / 2.0)
                else:
                    # Square grid physical coordinates
                    x_offset = c * grid_size
                    y_offset = -r * grid_size
                    
                new_pos[comp['indices'], 0] = comp['shifted_pos'][:, 0] + x_offset + center_x_offset
                new_pos[comp['indices'], 1] = comp['shifted_pos'][:, 1] + y_offset + center_y_offset
            break
        else:
            # Increase grid size and retry
            S = int(S * 1.1) + 2
            
    # --- 5. Center the final visualization ---
    global_min = np.min(new_pos, axis=0)
    global_max = np.max(new_pos, axis=0)
    center = (global_max + global_min) / 2.0
    new_pos -= center
    
    new_box_limit = max(global_max[0] - global_min[0], global_max[1] - global_min[1]) / 2.0 * 1.1
    
    print(f"Packed {len(components)} objects into a uniform grid. Ready for display.")
    return new_pos, new_box_limit

# --- 3. Main Layout Algorithm ---

def calculate_layout(connectivity, n_nodes, params):
    """
    Main layout generation pipeline.
    
    connectivity: N x 3 NumPy array representing [Source_Index, Target_Index, Score]
    n_nodes: Total number of nodes in the network
    params: Dictionary containing physics and execution parameters
    
    Returns:
        pos (np.ndarray): Final X/Y coordinates
        box_limit (float): Boundary box size
    """
    # Determine execution device
    use_gpu = False
    device_name = "CPU"
    if HAS_TORCH:
        device = Hardware_Utils.get_optimal_device()
        if device.type != "cpu":
            use_gpu = True
            device_name = f"GPU ({device})"
            
    print(f"Running layout calculation on {device_name}")
    
    edges = connectivity[:, :2].astype(np.int32)
    edge_scores = connectivity[:, 2]
    
    # Initialize basic grid positioning to start
    side = int(np.ceil(np.sqrt(n_nodes)))
    base_box = np.sqrt(n_nodes) * 2.5 + 5.0
    initial_box_limit = base_box * params.get('BOX_SCALE', 1.0)
    x = np.linspace(-initial_box_limit*0.5, initial_box_limit*0.5, side)
    y = np.linspace(-initial_box_limit*0.5, initial_box_limit*0.5, side)
    xv, yv = np.meshgrid(x, y)
    initial_pos = np.column_stack((xv.flatten(), yv.flatten()))[:n_nodes].astype(np.float32)

    components = find_connected_components(n_nodes, edges)
    
    # 1. Sort components from largest to smallest
    components.sort(key=len, reverse=True)
    
    # 2. Skip single nodes completely
    active_comps = [c for c in components if len(c) > 1]
    singletons = len(components) - len(active_comps)
    
    large_comps = [c for c in active_comps if len(c) >= 500]
    small_comps = [c for c in active_comps if len(c) < 500]

    batches = []
    current_batch = []
    current_nodes = 0
    BATCH_LIMIT = 2000

    for comp in small_comps:
        if current_nodes + len(comp) > BATCH_LIMIT and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_nodes = 0
        current_batch.append(comp)
        current_nodes += len(comp)
    if current_batch:
        batches.append(current_batch)

    jobs = [[c] for c in large_comps] + batches
    
    print(f"Found {len(active_comps)} active components.")
    print(f"  > Simulating {len(large_comps)} massive components individually.")
    print(f"  > Grouped {len(small_comps)} small components into {len(batches)} parallel batches (Max {BATCH_LIMIT} nodes/batch).")
    print(f"  > Skipped {singletons} single nodes.")
    
    final_pos = np.copy(initial_pos)
    
    # Pre-map edges and scores to components for O(1) extraction
    node_to_comp_idx = {}
    for c_idx, comp in enumerate(active_comps):
        for node in comp:
            node_to_comp_idx[node] = c_idx
            
    comp_edges = {c_idx: [] for c_idx in range(len(active_comps))}
    comp_scores = {c_idx: [] for c_idx in range(len(active_comps))}
    
    for i, (u, v) in enumerate(edges):
        if u in node_to_comp_idx: 
            c_idx = node_to_comp_idx[u]
            comp_edges[c_idx].append((u, v))
            comp_scores[c_idx].append(edge_scores[i])
    
    # 3. Simulate jobs sequentially
    for job_idx, batch_comps in enumerate(jobs):
        n_batch_nodes = sum(len(c) for c in batch_comps)
        is_large_job = len(batch_comps) == 1 and n_batch_nodes >= 500

        # Build batch-level arrays
        batch_global_nodes = []
        for c in batch_comps:
            batch_global_nodes.extend(c)

        global_to_batch = {g_id: l_id for l_id, g_id in enumerate(batch_global_nodes)}

        batch_edges_list = []
        batch_scores_list = []
        batch_pos_list = []

        grid_side = int(np.ceil(np.sqrt(len(batch_comps))))
        spacing = 100.0

        # Construct initial positions per component
        for c_idx_in_batch, c in enumerate(batch_comps):
            n_comp_nodes = len(c)
            c_idx = node_to_comp_idx[c[0]]

            c_edges = comp_edges[c_idx]
            c_scores = comp_scores[c_idx]

            comp_global_to_local = {g: l for l, g in enumerate(c)}
            c_local_edges = [(comp_global_to_local[u], comp_global_to_local[v]) for u, v in c_edges]

            comp_box_limit = (np.sqrt(n_comp_nodes) * 2.5 + 5.0) * params.get('BOX_SCALE', 1.0)
            local_pos = None
            spectral_success = False

            if n_comp_nodes >= 4:
                if n_comp_nodes >= 50:
                    print(f"  > Calculating Spectral Layout for sub-component ({n_comp_nodes} nodes)...")
                try:
                    import scipy.sparse as sp
                    from scipy.sparse.csgraph import laplacian
                    from scipy.sparse.linalg import eigsh

                    row = [e[0] for e in c_local_edges] + [e[1] for e in c_local_edges]
                    col = [e[1] for e in c_local_edges] + [e[0] for e in c_local_edges]
                    data = c_scores + c_scores
                    adj = sp.coo_matrix((data, (row, col)), shape=(n_comp_nodes, n_comp_nodes))

                    L = laplacian(adj, normed=True)
                    vals, vecs = eigsh(L, k=3, which='SM', tol=1e-3)

                    x_coords = vecs[:, 1]
                    y_coords = vecs[:, 2]

                    x_norm = (x_coords - np.min(x_coords)) / (np.ptp(x_coords) + 1e-9)
                    y_norm = (y_coords - np.min(y_coords)) / (np.ptp(y_coords) + 1e-9)

                    x_scaled = (x_norm - 0.5) * comp_box_limit * 0.8
                    y_scaled = (y_norm - 0.5) * comp_box_limit * 0.8

                    local_pos = np.column_stack((x_scaled, y_scaled)).astype(np.float32)
                    spectral_success = True
                except Exception as e:
                    if n_comp_nodes >= 50:
                        print(f"  > Spectral solver failed: {e}. Falling back to grid layout.")

            if not spectral_success:
                side_comp = int(np.ceil(np.sqrt(n_comp_nodes)))
                x_c = np.linspace(-comp_box_limit * 0.5, comp_box_limit * 0.5, side_comp)
                y_c = np.linspace(-comp_box_limit * 0.5, comp_box_limit * 0.5, side_comp)
                xv_c, yv_c = np.meshgrid(x_c, y_c)
                local_pos = np.column_stack((xv_c.flatten(), yv_c.flatten()))[:n_comp_nodes].astype(np.float32)

            row_grid = c_idx_in_batch // grid_side
            col_grid = c_idx_in_batch % grid_side
            offset_x = (col_grid - grid_side / 2.0) * spacing
            offset_y = (row_grid - grid_side / 2.0) * spacing
            
            local_pos[:, 0] += offset_x
            local_pos[:, 1] += offset_y

            batch_pos_list.append(local_pos)

            for (u, v), score in zip(c_edges, c_scores):
                batch_edges_list.append((global_to_batch[u], global_to_batch[v]))
                batch_scores_list.append(score)

        # Unify the arrays for the batch
        batch_pos = np.vstack(batch_pos_list).astype(np.float32)
        batch_box_limit = (np.sqrt(n_batch_nodes) * 2.5 + 5.0) * params.get('BOX_SCALE', 1.0)
        batch_comp_labels = np.zeros(n_batch_nodes, dtype=np.int32)

        batch_pos += np.random.normal(0, 0.1, batch_pos.shape).astype(np.float32)

        if is_large_job:
             print(f"\nSimulating Large Component {job_idx+1}/{len(jobs)} ({n_batch_nodes} nodes)...")
        else:
             print(f"\nSimulating Batch {job_idx+1}/{len(jobs)} ({len(batch_comps)} components, {n_batch_nodes} nodes)...")

        cutoffs = [params.get('SIMILARITY_THRESHOLD', 0.0)]
        if is_large_job and params.get('ENABLE_PROGRESSIVE_SIMULATION', True) and n_batch_nodes > 2000 and len(batch_scores_list) > 10:
            sorted_local = np.sort(batch_scores_list)[::-1] 
            n_edges = len(sorted_local)
            fractions = [0.2, 0.4, 0.6, 0.8]
            indices = [max(0, min(int(n_edges * f) - 1, n_edges - 1)) for f in fractions]
            raw_cutoffs = [sorted_local[i] for i in indices]
            
            cutoffs = []
            for c in raw_cutoffs:
                if not cutoffs or c < cutoffs[-1]:
                    cutoffs.append(c)
                    
            if not cutoffs or cutoffs[-1] > params.get('SIMILARITY_THRESHOLD', 0.0):
                cutoffs.append(params.get('SIMILARITY_THRESHOLD', 0.0))
            else:
                cutoffs[-1] = params.get('SIMILARITY_THRESHOLD', 0.0)
                
            print(f"  > Massive component detected. Using {len(cutoffs)}-stage progressive annealing (Edge-based).")
        
        for stage, cutoff in enumerate(cutoffs):
            if len(cutoffs) > 1:
                stage_edge_count = sum(1 for s in batch_scores_list if s >= cutoff)
                print(f"  > Stage {stage+1}/{len(cutoffs)}: Cutoff = {cutoff:.3f} | Active Edges: {stage_edge_count}")

            stage_edges = [edge for edge, score in zip(batch_edges_list, batch_scores_list) if score >= cutoff]

            if len(stage_edges) > 0:
                local_edges = np.array(stage_edges, dtype=np.int32)
            else:
                local_edges = np.zeros((0, 2), dtype=np.int32)
                
            if use_gpu:
                sim = SSNSimulationGPU(batch_pos, local_edges, batch_comp_labels, batch_box_limit, params)
            else:
                sim = SSNSimulationCPU(batch_pos, local_edges, batch_comp_labels, batch_box_limit, params)
                
            rmsd_window = params.get('RMSD_WINDOW', 50)
            max_steps = params.get('MAX_STEPS', 2000)
            rmsd_buffer = deque(maxlen=rmsd_window)
            avg_history = []
            
            for step in range(max_steps):
                rmsd = sim.step(step, apply_warmup=(stage == 0))
                rmsd_buffer.append(rmsd)
                avg_rmsd = np.mean(rmsd_buffer)
                
                if step > 0 and step % 500 == 0:
                    print(f"    - Step {step:04d}/{max_steps}: RMSD = {avg_rmsd:.5f}")
                    
                if len(rmsd_buffer) == rmsd_window:
                    avg_history.append(avg_rmsd)
                    
                    if avg_rmsd < params.get('RMSD_THRESHOLD', 0.005):
                        print(f"    - Converged at Step {step} (RMSD: {avg_rmsd:.5f})")
                        break
                        
                    pct_threshold = params.get('PERCENTAGE_DROP_THRESHOLD', 0.0)
                    warmup_steps = max_steps / 4.0
                    trend_window = 10
                    
                    if pct_threshold > 0.0 and len(avg_history) >= (rmsd_window + trend_window) and step > warmup_steps:
                        current_trend = np.mean(avg_history[-trend_window:])
                        old_trend = np.mean(avg_history[-(rmsd_window + trend_window):-rmsd_window])
                        
                        if old_trend > 0:
                            pct_drop = ((old_trend - current_trend) / old_trend) * 100.0
                            if pct_drop < pct_threshold:
                                print(f"    - Plateau Reached at Step {step} (Drop: {pct_drop:.3f}% < {pct_threshold}%)")
                                break
                    
            batch_pos = sim.get_pos()
            
            del sim
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # Update the final positions
        final_pos[batch_global_nodes] = batch_pos
            
    print("\nSimulation Complete.")
    
    # Pack independent components into a grid
    final_pos, final_box_limit = pack_components_to_grid(
        final_pos, edges, n_nodes, 
        params.get('PACKING_GRID_SIZE', 200.0), 
        params.get('PACKING_PADDING', 50.0),
        params.get('PACKING_GEOMETRY', 'Square')
    )
    
    return final_pos, final_box_limit
