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
    from utilities import Hardware_Utils
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

# --- SGLD / Monte Carlo Parameters (Defaults defined here as requested) ---
SGLD_NEGATIVE_SAMPLES = 20     # Number of random negative (repulsive) pairs sampled per node
SGLD_NOISE_SCALE = 1.0         # Global scalar for thermal noise perturbation
SGLD_START_TEMP = 1.5          # Starting SGLD temperature (kinetic heat) in Stage 1


# --- 1. Physics Kernels ---

def _get_physics_kernel_sgld():
    def _run_physics_kernel_sgld(pos, vel, springs, box_limit, dt, damping, k_spr, k_coul, max_f, temperature, neg_samples_K, cutoff_dist):
        n_balls = pos.shape[0]
        acc = np.zeros_like(pos)
        
        # --- SPRINGS (Attraction) ---
        for i in range(springs.shape[0]):
            idx_a, idx_b = springs[i, 0], springs[i, 1]
            dx = pos[idx_a, 0] - pos[idx_b, 0]
            dy = pos[idx_a, 1] - pos[idx_b, 1]
            
            # Force gradient: -k_spr * distance * direction_vector
            acc[idx_a, 0] -= k_spr * dx
            acc[idx_a, 1] -= k_spr * dy
            acc[idx_b, 0] += k_spr * dx
            acc[idx_b, 1] += k_spr * dy
            
        # --- REPULSION (Negative Sampling) ---
        if n_balls > 1:
            scale_factor = (n_balls - 1.0) / neg_samples_K
            cutoff_sq = cutoff_dist * cutoff_dist
            for i in range(n_balls):
                rep_x = 0.0
                rep_y = 0.0
                for k in range(neg_samples_K):
                    # Randomly sample a negative neighbor
                    j = np.random.randint(0, n_balls)
                    if j == i:
                        continue
                        
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist_sq = dx*dx + dy*dy
                    
                    if dist_sq > cutoff_sq:
                        continue
                        
                    dist = math.sqrt(dist_sq) + 1e-9
                    safe_dist = max(dist, 0.5)
                    f = k_coul / (safe_dist * safe_dist)
                    if f > max_f:
                        f = max_f
                        
                    rep_x += f * (dx / dist)
                    rep_y += f * (dy / dist)
                acc[i, 0] += rep_x * scale_factor
                acc[i, 1] += rep_y * scale_factor
                    
        # --- INTEGRATION (Underdamped Langevin Dynamics) ---
        rmsd = 0.0
        # Thermal velocity noise scale: sqrt(2 * damping * temperature * dt)
        noise_scale = math.sqrt(2.0 * damping * temperature * dt) if temperature > 0.0 else 0.0
        
        for i in range(n_balls):
            # Friction / Damping
            acc[i, 0] -= damping * vel[i, 0]
            acc[i, 1] -= damping * vel[i, 1]
            
            # Velocity update
            vel[i, 0] += acc[i, 0] * dt
            vel[i, 1] += acc[i, 1] * dt
            
            # Inject thermal kinetic noise (MCMC step)
            if noise_scale > 0.0:
                vel[i, 0] += np.random.normal(0, 1.0) * noise_scale
                vel[i, 1] += np.random.normal(0, 1.0) * noise_scale
                
            old_x, old_y = pos[i, 0], pos[i, 1]
            
            pos[i, 0] += vel[i, 0] * dt
            pos[i, 1] += vel[i, 1] * dt
            
            # Bouncing walls
            if pos[i, 0] > box_limit:
                pos[i, 0] = box_limit
                vel[i, 0] *= -0.5
            elif pos[i, 0] < -box_limit:
                pos[i, 0] = -box_limit
                vel[i, 0] *= -0.5
                
            if pos[i, 1] > box_limit:
                pos[i, 1] = box_limit
                vel[i, 1] *= -0.5
            elif pos[i, 1] < -box_limit:
                pos[i, 1] = -box_limit
                vel[i, 1] *= -0.5
                
            diff_x = pos[i, 0] - old_x
            diff_y = pos[i, 1] - old_y
            rmsd += diff_x*diff_x + diff_y*diff_y
            
        return math.sqrt(rmsd / n_balls)

    if NUMBA_AVAILABLE:
        return jit(nopython=True, fastmath=True)(_run_physics_kernel_sgld)
    return _run_physics_kernel_sgld

run_physics_kernel_sgld = _get_physics_kernel_sgld()


class SSNSimulationCPU:
    def __init__(self, pos, springs, comp_labels, box_limit, params):
        self.pos = pos.astype(np.float32)
        self.vel = np.zeros_like(pos)
        self.springs = springs
        self.comp_labels = comp_labels
        self.box = box_limit
        self.params = params
        self.last_rmsd = 0.0
        
    def step(self, current_step, apply_warmup=True):
        max_steps = self.params.get('MAX_STEPS', 2000)
        
        start_temp = self.params.get('SGLD_START_TEMP', SGLD_START_TEMP)
        noise_scale = self.params.get('SGLD_NOISE_SCALE', SGLD_NOISE_SCALE)
        
        # SGLD Temperature Annealing schedule with Thermal Quenching
        if apply_warmup:
            temperature = start_temp
        else:
            progress = current_step / max(1.0, float(max_steps))
            if progress < 0.5:
                temperature = start_temp * (1.0 - progress / 0.5)
            else:
                temperature = 0.0
            
        temperature = max(0.0, temperature) * noise_scale
        
        # Calculate RMSD
        sgld_k = self.params.get('SGLD_K', SGLD_NEGATIVE_SAMPLES)
        cutoff_dist = self.params.get('COULOMB_CUTOFF', 30.0)
        self.last_rmsd = run_physics_kernel_sgld(
            self.pos, self.vel, self.springs, self.box,
            self.params.get('DT', 0.1),
            self.params.get('DAMPING', 0.5),
            self.params.get('SPRING_K', 0.1),
            self.params.get('COULOMB_K', 50.0),
            self.params.get('MAX_FORCE_LIMIT', 10.0),
            temperature,
            sgld_k,
            cutoff_dist
        )
        return self.last_rmsd
        
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
            self.last_rmsd = 0.0
        
        @torch.no_grad()
        def step(self, current_step, apply_warmup=True):
            max_steps = self.params.get('MAX_STEPS', 2000)
            
            start_temp = self.params.get('SGLD_START_TEMP', SGLD_START_TEMP)
            noise_scale = self.params.get('SGLD_NOISE_SCALE', SGLD_NOISE_SCALE)
            
            # --- 1. SGLD Temperature Annealing schedule with Thermal Quenching ---
            if apply_warmup:
                temperature = start_temp
            else:
                progress = current_step / max(1.0, float(max_steps))
                if progress < 0.5:
                    temperature = start_temp * (1.0 - progress / 0.5)
                else:
                    temperature = 0.0
                
            temperature = max(0.0, temperature) * noise_scale
            
            acc = torch.zeros_like(self.pos)
            N = self.pos.size(0)
            
            # --- 2. ATTRACTION (Spring Forces) ---
            if len(self.springs) > 0:
                idx_a, idx_b = self.springs[:, 0], self.springs[:, 1]
                pa, pb = self.pos[idx_a], self.pos[idx_b]
                
                spring_k = self.params.get('SPRING_K', 0.1)
                fv = -spring_k * (pa - pb)
                
                acc.index_add_(0, idx_a, fv)
                acc.index_add_(0, idx_b, -fv)
                
                del pa, pb, fv
                
            # --- 3. REPULSION (Negative Sampling on GPU) ---
            k_coul = self.params.get('COULOMB_K', 50.0)
            max_f = self.params.get('MAX_FORCE_LIMIT', 10.0)
            cutoff_dist = self.params.get('COULOMB_CUTOFF', 30.0)
            
            if N > 1:
                sgld_k = self.params.get('SGLD_K', SGLD_NEGATIVE_SAMPLES)
                # Sample K random indices for each node in the component
                neg_nodes = torch.randint(0, N, (N, sgld_k), device=self.device)
                
                # Reshape to compute pairwise distances
                pos_expanded = self.pos.unsqueeze(1)    # Shape: [N, 1, 2]
                neg_pos = self.pos[neg_nodes]            # Shape: [N, sgld_k, 2]
                
                delta = pos_expanded - neg_pos           # Shape: [N, sgld_k, 2]
                dist = delta.norm(dim=2) + 1e-9          # Shape: [N, sgld_k]
                
                # Repulsion magnitude: f = k_coul / max(dist, 0.5)^2
                safe_dist = torch.clamp(dist, min=0.5)
                f_mag = k_coul / (safe_dist ** 2)
                f_mag = torch.clamp(f_mag, max=max_f)
                
                # Mask out self-repulsion and nodes beyond COULOMB_CUTOFF using torch.where
                is_self = neg_nodes == torch.arange(N, device=self.device).unsqueeze(1)
                is_far = dist > cutoff_dist
                cond = ~(is_self | is_far)
                f_mag = torch.where(cond, f_mag, 0.0)
                
                # Force vector
                f_vec = (f_mag / dist).unsqueeze(2) * delta  # Shape: [N, sgld_k, 2]
                
                # Accumulate over K negative samples and scale by Monte Carlo estimator (N-1)/K
                scale_factor = float(N - 1) / float(sgld_k)
                acc += f_vec.sum(dim=1) * scale_factor
                
                del neg_nodes, pos_expanded, neg_pos, delta, dist, safe_dist, f_mag, f_vec
                
            # --- 4. INTEGRATION (Underdamped Langevin Dynamics) ---
            damping = self.params.get('DAMPING', 0.5)
            dt = self.params.get('DT', 0.1)
            
            # Apply friction
            acc -= damping * self.vel
            self.vel += acc * dt
            
            # Add thermal Langevin noise to velocity
            if temperature > 0.0:
                noise_scale = math.sqrt(2.0 * damping * temperature * dt)
                noise = torch.randn_like(self.vel) * noise_scale
                self.vel += noise
                
            old = self.pos.clone()
            self.pos += self.vel * dt
            
            # Boundary collisions (Bouncing)
            out_of_bounds_x = self.pos[:, 0].abs() > self.box
            out_of_bounds_y = self.pos[:, 1].abs() > self.box
            self.vel[out_of_bounds_x, 0] *= -0.5
            self.vel[out_of_bounds_y, 1] *= -0.5
            
            self.pos.clamp_(min=-self.box, max=self.box)
            
            # Transfer RMSD to host once every 10 steps to prevent PCIe stalls
            if (current_step % 10 == 0) or (current_step == max_steps - 1):
                self.last_rmsd = (self.pos - old).norm(dim=1).pow(2).mean().sqrt().item()
                
            del acc, old
            return self.last_rmsd

        def get_pos(self): return self.pos.cpu().numpy()


# --- 2. Components & Packing Logic (Identical API for seamless integration) ---

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
    """Maps each node to its connected component ID."""
    components = find_connected_components(n_nodes, edges)
    labels = np.zeros(n_nodes, dtype=np.int32)
    for c_id, comp in enumerate(components):
        for node in comp:
            labels[node] = c_id
    return labels


def pack_components_to_grid(pos, edges, n_nodes, grid_size, padding):
    """Packs independent components into a strict grid layout."""
    components = find_connected_components(n_nodes, edges)
    if not components:
        return pos, 100.0

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
        
        min_x = np.min(comp_pos[:, 0])
        max_y = np.max(comp_pos[:, 1]) 
        
        shifted_pos = comp_pos - [min_x, max_y]
        global_to_local = {g: l for l, g in enumerate(comp)}
        points = list(shifted_pos)
        
        for u, v in comp_edges[c_id]:
            p1 = shifted_pos[global_to_local[u]]
            p2 = shifted_pos[global_to_local[v]]
            dist = np.hypot(p2[0]-p1[0], p2[1]-p1[1])
            steps = int(dist / (grid_size / 4)) + 1
            for i in range(1, steps):
                t = i / steps
                px = p1[0] + t * (p2[0] - p1[0])
                py = p1[1] + t * (p2[1] - p1[1])
                points.append([px, py])
                
        max_c, max_r = 0, 0
        pad = padding / 2.0
        
        for px, py in points:
            pos_y = -py
            c_max = int((px + pad) / grid_size)
            r_max = int((pos_y + pad) / grid_size)
            max_c = max(max_c, c_max)
            max_r = max(max_r, r_max)
            
        cols = max_c + 1
        rows = max_r + 1
        mask = np.zeros((rows, cols), dtype=bool)
        
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
        
    comp_info.sort(key=lambda x: (x['area'], x['num_nodes']), reverse=True)
    
    total_area = sum(c['area'] for c in comp_info)
    max_comp_cols = max(c['cols'] for c in comp_info)
    target_width_grids = max(int(math.ceil(math.sqrt(total_area) * 1.5)), max_comp_cols)
    target_height_grids = total_area + max(c['rows'] for c in comp_info) + 10
    
    grid_map = np.zeros((target_height_grids, target_width_grids), dtype=bool)
    new_pos = np.zeros((n_nodes, 2), dtype=np.float32)
    new_pos[:] = pos[:]
    
    center_x_offset = grid_size / 2.0  
    center_y_offset = -grid_size / 2.0 
    
    for comp in comp_info:
        mask = comp['mask']
        h, w = comp['rows'], comp['cols']
        placed = False
        
        for r in range(grid_map.shape[0] - h + 1):
            for c in range(grid_map.shape[1] - w + 1):
                if not np.any(grid_map[r:r+h, c:c+w] & mask):
                    grid_map[r:r+h, c:c+w] |= mask
                    
                    x_offset = c * grid_size
                    y_offset = -r * grid_size
                    
                    new_pos[comp['indices'], 0] = comp['shifted_pos'][:, 0] + x_offset + center_x_offset
                    new_pos[comp['indices'], 1] = comp['shifted_pos'][:, 1] + y_offset + center_y_offset
                    placed = True
                    break
            if placed:
                break
                
    global_min = np.min(new_pos, axis=0)
    global_max = np.max(new_pos, axis=0)
    center = (global_max + global_min) / 2.0
    new_pos -= center
    
    new_box_limit = max(global_max[0] - global_min[0], global_max[1] - global_min[1]) / 2.0 * 1.1
    return new_pos, new_box_limit


# --- 3. Main Layout Entrypoint ---

def calculate_layout(connectivity, n_nodes, params):
    """
    Main layout generation pipeline using Monte Carlo SGLD.
    """
    edges = connectivity[:, :2].astype(np.int32)
    edge_scores = connectivity[:, 2]
    
    # Initialize basic grid positioning
    print("Computing initial node positions using Laplacian Spectral / Grid layouts...")
    side = int(np.ceil(np.sqrt(n_nodes)))
    base_box = np.sqrt(n_nodes) * 2.5 + 5.0
    initial_box_limit = base_box * params.get('BOX_SCALE', 1.0)
    x = np.linspace(-initial_box_limit*0.5, initial_box_limit*0.5, side)
    y = np.linspace(-initial_box_limit*0.5, initial_box_limit*0.5, side)
    xv, yv = np.meshgrid(x, y)
    initial_pos = np.column_stack((xv.flatten(), yv.flatten()))[:n_nodes].astype(np.float32)

    components = find_connected_components(n_nodes, edges)
    components.sort(key=len, reverse=True)
    
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
    print(f"  > SGLD solver simulating {len(large_comps)} large components individually.")
    print(f"  > Grouped {len(small_comps)} small components into {len(batches)} parallel batches.")
    
    final_pos = np.copy(initial_pos)
    
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
            
    # Simulate jobs sequentially
    for job_idx, batch_comps in enumerate(jobs):
        n_batch_nodes = sum(len(c) for c in batch_comps)
        is_large_job = len(batch_comps) == 1 and n_batch_nodes >= 500

        batch_global_nodes = []
        for c in batch_comps:
            batch_global_nodes.extend(c)

        global_to_batch = {g_id: l_id for l_id, g_id in enumerate(batch_global_nodes)}

        batch_edges_list = []
        batch_scores_list = []
        batch_pos_list = []

        grid_side = int(np.ceil(np.sqrt(len(batch_comps))))
        spacing = 100.0

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

        batch_pos = np.vstack(batch_pos_list).astype(np.float32)
        batch_box_limit = (np.sqrt(n_batch_nodes) * 2.5 + 5.0) * params.get('BOX_SCALE', 1.0)
        batch_comp_labels = np.zeros(n_batch_nodes, dtype=np.int32)
        batch_pos += np.random.normal(0, 0.1, batch_pos.shape).astype(np.float32)

        # Calculate dynamic K based on batch size: max(SGLD_MIN_K, int(SGLD_K_PERCENT * N))
        batch_params = params.copy()
        min_k = batch_params.get('SGLD_MIN_K', 20)
        pct_k = batch_params.get('SGLD_K_PERCENT', 0.01)
        batch_params['SGLD_K'] = max(min_k, int(pct_k * n_batch_nodes))

        cutoffs = [batch_params.get('SIMILARITY_THRESHOLD', 0.0)]
        if is_large_job and batch_params.get('ENABLE_PROGRESSIVE_SIMULATION', True) and n_batch_nodes > 2000 and len(batch_scores_list) > 10:
            sorted_local = np.sort(batch_scores_list)[::-1] 
            n_edges = len(sorted_local)
            fractions = [0.2, 0.4, 0.6, 0.8]
            indices = [max(0, min(int(n_edges * f) - 1, n_edges - 1)) for f in fractions]
            raw_cutoffs = [sorted_local[i] for i in indices]
            
            cutoffs = []
            for c in raw_cutoffs:
                if not cutoffs or c < cutoffs[-1]:
                    cutoffs.append(c)
                    
            if not cutoffs or cutoffs[-1] > batch_params.get('SIMILARITY_THRESHOLD', 0.0):
                cutoffs.append(batch_params.get('SIMILARITY_THRESHOLD', 0.0))
            else:
                cutoffs[-1] = batch_params.get('SIMILARITY_THRESHOLD', 0.0)
                
            print(f"  > Progressive SGLD initialized with {len(cutoffs)} stages.")
        
        for stage, cutoff in enumerate(cutoffs):
            stage_edges = [edge for edge, score in zip(batch_edges_list, batch_scores_list) if score >= cutoff]
            if len(stage_edges) > 0:
                local_edges = np.array(stage_edges, dtype=np.int32)
            else:
                local_edges = np.zeros((0, 2), dtype=np.int32)
                
            if HAS_TORCH and torch.cuda.is_available():
                sim = SSNSimulationGPU(batch_pos, local_edges, batch_comp_labels, batch_box_limit, batch_params)
            else:
                sim = SSNSimulationCPU(batch_pos, local_edges, batch_comp_labels, batch_box_limit, batch_params)
                
            rmsd_window = batch_params.get('RMSD_WINDOW', 50)
            max_steps = batch_params.get('MAX_STEPS', 2000)
            rmsd_buffer = deque(maxlen=rmsd_window)
            avg_history = []
            
            for step in range(max_steps):
                rmsd = sim.step(step, apply_warmup=(stage == 0))
                rmsd_buffer.append(rmsd)
                avg_rmsd = np.mean(rmsd_buffer)
                
                 # Progress reporting every 500 steps
                if step > 0 and step % 500 == 0:
                    print(f"    - Step {step:04d}/{max_steps} | RMSD: {avg_rmsd:.5f}")
                
                if len(rmsd_buffer) == rmsd_window:
                    avg_history.append(avg_rmsd)
                    
                    progress = step / float(max_steps)
                    if avg_rmsd < batch_params.get('RMSD_THRESHOLD', 0.005) and progress > 0.8:
                        print(f"    - Converged at Step {step} (RMSD: {avg_rmsd:.5f})")
                        break
                        
                    pct_threshold = batch_params.get('PERCENTAGE_DROP_THRESHOLD', 0.0)
                    warmup_steps = max_steps / 4.0
                    trend_window = 10
                    
                    if pct_threshold > 0.0 and len(avg_history) >= (rmsd_window + trend_window) and step > warmup_steps:
                        current_trend = np.mean(avg_history[-trend_window:])
                        old_trend = np.mean(avg_history[-(rmsd_window + trend_window):-rmsd_window])
                        
                        if old_trend > 0:
                            pct_drop = ((old_trend - current_trend) / old_trend) * 100.0
                            if pct_drop < pct_threshold and progress > 0.8:
                                print(f"    - Plateau Reached at Step {step} (Drop: {pct_drop:.3f}% < {pct_threshold}%)")
                                break
                    
            batch_pos = sim.get_pos()
            del sim
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        final_pos[batch_global_nodes] = batch_pos
            
    # Pack independent components into a grid
    final_pos, final_box_limit = pack_components_to_grid(
        final_pos, edges, n_nodes, 
        params.get('PACKING_GRID_SIZE', 200.0), 
        params.get('PACKING_PADDING', 50.0)
    )
    return final_pos, final_box_limit
