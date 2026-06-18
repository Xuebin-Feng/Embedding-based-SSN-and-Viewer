import Command_Engine
import numpy as np
import SSN_Config as cfg

def run(viewer, args):
    if args and args[0].lower() in ['help', '-h', '--help']:
        msg = ("Usage: hide [single/free]\n\n"
               "Description:\n"
               "  Without arguments: Immediately hides all currently selected nodes and their connected edges.\n"
               "  With 'single' or 'free': Hides all visible nodes that have no active edges at the current similarity threshold.\n\n"
               "To unhide nodes, use the `reset hide` command.")
        Command_Engine.print_help(viewer, msg)
        return

    if args and args[0].lower() == 'reset':
        Command_Engine.execute_reset(viewer, ["hidden"])
        return

    if args and args[0].lower() in ['single', 'free']:
        current_slider_val = getattr(viewer, 'current_slider_threshold', getattr(cfg, 'SIMILARITY_THRESHOLD', 0.0))
        
        # Calculate which edges are active (visible endpoints and score >= threshold)
        nodes_visible_mask = viewer.visible_mask[viewer.edges[:, 0]] & viewer.visible_mask[viewer.edges[:, 1]]
        if hasattr(viewer, 'edge_scores') and len(viewer.edge_scores) > 0:
            threshold_visible_mask = viewer.edge_scores >= current_slider_val
            valid_edges_mask = nodes_visible_mask & threshold_visible_mask
        else:
            valid_edges_mask = nodes_visible_mask
            
        active_edges = viewer.edges[valid_edges_mask]
        
        # Find which nodes are endpoints of active edges
        has_active_edges = np.zeros(viewer.n_nodes, dtype=bool)
        if len(active_edges) > 0:
            has_active_edges[active_edges[:, 0]] = True
            has_active_edges[active_edges[:, 1]] = True
            
        # Select currently visible nodes that have no active edges
        single_nodes_mask = viewer.visible_mask & ~has_active_edges
        num_hidden = np.sum(single_nodes_mask)
        
        if num_hidden == 0:
            msg = "No single/free nodes found to hide at the current edge threshold."
            Command_Engine.print_help(viewer, msg)
            return
            
        viewer._save_state()
        viewer.visible_mask[single_nodes_mask] = False
        
        # Clean up selection if any selected nodes were hidden
        if hasattr(viewer, 'selected_indices'):
            viewer.selected_indices = [i for i in viewer.selected_indices if viewer.visible_mask[i]]
            
        viewer.hovered_node_idx = None
        viewer.selected_node_idx = None
        viewer.tooltip.text = ""
        
        viewer.update_selection_visual()
        viewer.update_edges()
        
        msg = f"Hidden {num_hidden} single/free nodes."
        Command_Engine.print_help(viewer, msg)
        return

    if not getattr(viewer, 'selected_indices', []):
        msg = "Error: No nodes currently selected."
        Command_Engine.print_help(viewer, msg)
        return
    
    viewer._save_state()
    viewer.visible_mask[viewer.selected_indices] = False
    viewer.selected_indices = []
    
    viewer.hovered_node_idx = None
    viewer.selected_node_idx = None
    viewer.tooltip.text = ""
    
    viewer.update_selection_visual()
    viewer.update_edges()
    
    msg = "Selected nodes hidden."
    Command_Engine.print_help(viewer, msg)
