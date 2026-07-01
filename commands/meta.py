import Command_Engine
import os
import re
import numpy as np
import pandas as pd
import h5py
import SSN_Utils as utils
from PyQt6 import QtWidgets

def print_help(meta_dir_help):
    print(f"""
    Node Metadata Manager Tool
    ==========================
    Usage:
      meta
          Opens a file explorer to manually select a metadata file (.xlsx, .xls, .csv)
          to upload and merge into the current viewer session.
      meta display/show <property_name>
          Displays the selected property of a node in the top-right corner of the window
          whenever a node is left-clicked.
      meta display/show clear/off
          Clears and removes the metadata property display.
      meta retrieve/download/export [filename] [expression]
          Downloads the current session metadata. If no filename is provided, 
          opens a save file dialog for location and name selection.
          An optional boolean logic expression can be provided at the end to filter 
          which nodes are exported (e.g., #cluster_1#, {{Length>500}}, or $sele$).
      meta help
          Displays this help message.

    Examples:
      meta
      meta show Organism
      meta show clear
      meta download
      meta download {{Length>500}}
      meta export filtered_meta.xlsx #cluster_1#
      meta export selected_meta.xlsx $sele$
    """)

def is_logic_expression(arg):
    # Check if the argument has typical boolean logic expression characters:
    # { } (metadata), # # (cluster/labels), @ @ (file search), " " (header text query),
    # &, |, !, ^ (logic operators), or represents UI selection $sele$
    if any(c in arg for c in '{{}}#@&|!^"'):
        return True
    if arg.lower() == '$sele$':
        return True
    # check AA position format (e.g. P106, _100)
    if re.match(r'^[a-zA-Z_][\d\.]+$', arg):
        return True
    return False

def run(viewer, args):
    # --- 1. Help & Usage ---
    import SSN_Config as cfg
    meta_dir = getattr(cfg, 'METADATA_DIR', os.path.join("Cache_Files", "Meta_Data"))
    os.makedirs(meta_dir, exist_ok=True)

    if args and args[0].lower() in ['help', '-h', '--help']:
        print_help(meta_dir)
        if hasattr(viewer, 'console_text'):
            viewer.console_text.text = "Help information printed to the terminal"
        return

    # --- 1.5. Display/Show Property ---
    if args and args[0].lower() in ['display', 'show']:
        if len(args) < 2:
            msg = "Usage: meta display/show <property_name> OR meta display/show clear/off"
            Command_Engine.print_help(viewer, msg)
            return

        prop_name = " ".join(args[1:]).strip()
        if prop_name.lower() in ['clear', 'off']:
            if 'meta_display' in viewer.hud_displays:
                viewer.hud_displays['meta_display'].hide()
            viewer.meta_display_prop = None
            msg = "Metadata display cleared."
            Command_Engine.print_help(viewer, msg)
            return

        if not prop_name:
            msg = "Error: Please specify a valid property name."
            Command_Engine.print_help(viewer, msg)
            return

        # Case-insensitive check in metadata
        available_props = list(viewer.metadata.keys()) if getattr(viewer, 'metadata', None) else []
        resolved_prop = None
        for p in available_props:
            if p.lower() == prop_name.lower():
                resolved_prop = p
                break

        if not resolved_prop:
            resolved_prop = prop_name
            if available_props:
                print(f"Warning: Property '{prop_name}' not found in current metadata. Available properties: {', '.join(available_props)}")

        viewer.meta_display_prop = resolved_prop

        # Register HUD display if not already present
        if 'meta_display' not in viewer.hud_displays:
            from SSN_Viewer import HUDDisplay
            
            class MetaHUDDisplay(HUDDisplay):
                def __init__(self, main_viewer):
                    super().__init__(
                        viewer=main_viewer,
                        name='meta_display',
                        pos_fn=lambda size: (size[0] - 30, 60),
                        anchor_x='right',
                        anchor_y='bottom'
                    )
                
                def on_node_clicked(self, node_idx):
                    p_name = getattr(self.viewer, 'meta_display_prop', None)
                    if p_name and getattr(self.viewer, 'metadata', None) and p_name in self.viewer.metadata:
                        val = self.viewer.metadata[p_name]["values"][node_idx]
                        import pandas as pd
                        val_str = str(val).strip() if pd.notna(val) and val is not None else "N/A"
                        self.show(f"{p_name}: {val_str}")
                    else:
                        self.show(f"{p_name}: N/A")

            viewer.hud_displays['meta_display'] = MetaHUDDisplay(viewer)

        # Update if a node is currently selected
        display = viewer.hud_displays['meta_display']
        node_idx = getattr(viewer, 'selected_node_idx', None)
        if node_idx is not None and getattr(viewer, 'metadata', None) and resolved_prop in viewer.metadata:
            val = viewer.metadata[resolved_prop]["values"][node_idx]
            val_str = str(val).strip() if pd.notna(val) and val is not None else "N/A"
            display.show(f"{resolved_prop}: {val_str}")
        else:
            display.show(f"{resolved_prop}: -")

        msg = f"Metadata display enabled for property: '{resolved_prop}'"
        Command_Engine.print_help(viewer, msg)
        return

    # --- 2. Resolve Directory & Path ---
    is_download = False
    filepath = None
    filename = ""
    expr = None

    # Check if download mode is requested
    if args and args[0].lower() in ['retrieve', 'download', 'export']:
        is_download = True
        
        # Check if filename or expression is provided
        if len(args) >= 2:
            if len(args) >= 3 and is_logic_expression(args[1]) and not is_logic_expression(args[-1]):
                # Form: download <expression> <filename>
                filename = args[-1]
                expr = " ".join(args[1:-1]).strip()
            elif is_logic_expression(args[1]):
                # Form: download <expression> (filename will be selected via dialog)
                expr = " ".join(args[1:]).strip()
            else:
                # Form: download <filename> [expression]
                filename = args[1]
                if len(args) >= 3:
                    expr = " ".join(args[2:]).strip()

        if filename:
            # Enforce default .xlsx extension if no extension is specified
            _, ext = os.path.splitext(filename)
            if not ext:
                filename += ".xlsx"
                ext = ".xlsx"
            filepath = os.path.join(meta_dir, filename)
        else:
            # Open save file dialog if no filename provided for download
            try:
                file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                    viewer.canvas.native,
                    "Save Metadata File",
                    meta_dir,
                    "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)"
                )
                if not file_path:
                    msg = "Metadata export cancelled."
                    Command_Engine.print_help(viewer, msg)
                    return
                filepath = file_path
                filename = os.path.basename(file_path)
                _, ext = os.path.splitext(filename)
            except Exception as e:
                msg = f"Error opening save dialog: {e}"
                Command_Engine.print_help(viewer, msg)
                return
    else:
        # Upload mode
        if args:
            msg = "Error: Invalid arguments. To upload metadata, run 'meta' with no arguments to open the file explorer."
            Command_Engine.print_help(viewer, msg)
            return

        # No arguments -> Open file selection dialog (multiple files allowed)
        try:
            file_paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
                viewer.canvas.native,
                "Select Metadata Files",
                meta_dir,
                "Excel Files (*.xlsx *.xls);;CSV Files (*.csv);;All Files (*)"
            )
            if not file_paths:
                msg = "Metadata upload cancelled."
                Command_Engine.print_help(viewer, msg)
                return
        except Exception as e:
            msg = f"Error opening file dialog: {e}"
            Command_Engine.print_help(viewer, msg)
            return

    # --- 3. Execute Download Mode ---
    if is_download:
        if not getattr(viewer, 'metadata', None):
            msg = "Error: No metadata available in the viewer to download."
            Command_Engine.print_help(viewer, msg)
            return

        try:
            # Evaluate logic expression if provided
            mask = np.ones(viewer.n_nodes, dtype=bool)
            if expr:
                # Update the selection file in case the expression references $sele$
                header_dir = getattr(cfg, 'HEADER_LIST_DIR', os.path.join("Input_Files", "Header_Lists"))
                os.makedirs(header_dir, exist_ok=True)
                sele_path = os.path.join(header_dir, "_sele.txt")
                if hasattr(viewer, 'selected_indices') and viewer.selected_indices:
                    with open(sele_path, "w", encoding="utf-8") as f:
                        for idx in viewer.selected_indices:
                            f.write(viewer.full_headers[idx] + "\n")
                else:
                    if os.path.exists(sele_path):
                        open(sele_path, 'w').close()

                # Preprocess expression
                expr_cleaned = re.sub(r'["\']?\$sele\$["\']?', '@_sele.txt@', expr, flags=re.IGNORECASE)
                expr_cleaned = re.sub(r'\{([^}]+)\}', lambda m: '{' + m.group(1).replace(' ', '') + '}', expr_cleaned)
                
                viewer_to_aln = np.full(len(viewer.full_headers), -1, dtype=int)
                if (getattr(viewer, 'alignment', None).aln if getattr(viewer, 'alignment', None) else None) is not None:
                    for i, h in enumerate(viewer.full_headers):
                        if h in viewer.alignment.seq_map:
                            viewer_to_aln[i] = viewer.alignment.seq_map[h]
                valid_indices = np.where(viewer_to_aln != -1)[0]
                
                mask = Command_Engine.parse_advanced_expression(
                    expr_cleaned,
                    viewer_to_aln,
                    valid_indices,
                    viewer.full_headers,
                    getattr(viewer, 'cluster_labels', None),
                    getattr(viewer, 'group_labels', None),
                    getattr(viewer, 'alignment', None),
                    metadata=viewer.metadata
                )
                
                if np.sum(mask) == 0:
                    msg = f"Error: No nodes matched the expression '{expr}'."
                    Command_Engine.print_help(viewer, msg)
                    return

            # Build spreadsheet layout
            prop_names = [p for p in viewer.metadata.keys() if p != "Length"]
            
            row_0 = [""] + prop_names
            row_1 = [""] + [viewer.metadata[p]["type"] for p in prop_names]
            
            rows = [row_0, row_1]
            for i in range(viewer.n_nodes):
                if not mask[i]:
                    continue
                has_valid_prop = False
                row_val = [viewer.full_headers[i]]
                for p in prop_names:
                    val = viewer.metadata[p]["values"][i]
                    if viewer.metadata[p]["type"] == "number":
                        if pd.notna(val):
                            has_valid_prop = True
                            row_val.append(val)
                        else:
                            row_val.append("")
                    else:
                        if val is not None and str(val).strip() != "":
                            has_valid_prop = True
                            row_val.append(val)
                        else:
                            row_val.append("")
                
                # Only include nodes that have at least one populated property,
                # unless they were explicitly selected via expression filter.
                if has_valid_prop or expr:
                    rows.append(row_val)

            df = pd.DataFrame(rows)

            if ext.lower() == ".csv":
                df.to_csv(filepath, header=False, index=False)
            else:
                df.to_excel(filepath, header=False, index=False)

            msg = f"Metadata successfully downloaded to {filepath}"
            if expr:
                msg += f" (filtered by: {expr})"
            Command_Engine.print_help(viewer, msg)
        except Exception as e:
            msg = f"Error downloading metadata: {e}"
            Command_Engine.print_help(viewer, msg)
        return

    # --- 4. Execute Upload Mode ---
    successful_files = []
    failed_files = []
    matched_nodes = set()
    total_unmatched = 0
    all_merged_props = set()

    # Safe save state for undo support (save once before all uploads)
    viewer._save_state()

    for filepath in file_paths:
        filename = os.path.basename(filepath)
        _, ext = os.path.splitext(filename)
        if not os.path.exists(filepath):
            failed_files.append((filename, "File not found."))
            continue

        try:
            # Load the file
            if ext.lower() == ".csv":
                df = pd.read_csv(filepath, header=None, dtype=str)
            else:
                df = pd.read_excel(filepath, header=None)

            # Check dimensions
            if df.shape[0] < 3 or df.shape[1] < 2:
                raise ValueError("Invalid file format. Must contain at least sequence headers and one property column.")

            # Extract property names and types from first 2 rows (columns 1 onwards)
            prop_names = []
            valid_cols = []
            for col_idx in range(1, df.shape[1]):
                val = df.iloc[0, col_idx]
                if pd.notna(val) and str(val).strip():
                    prop_names.append(str(val).strip())
                    valid_cols.append(col_idx)

            if not prop_names:
                raise ValueError("No valid property names found in the first row.")

            # Validate property names to ensure they only contain alphanumeric characters, underscores, hyphens, and periods
            illegal_props = [prop for prop in prop_names if not re.match(r'^[a-zA-Z0-9_\-\.]+$', prop)]
            if illegal_props:
                raise ValueError(
                    f"Property names {', '.join([repr(p) for p in illegal_props])} contain illegal characters. "
                    "Allowed characters are: letters, numbers, underscores (_), hyphens (-), and periods (.)"
                )

            prop_types = []
            for col_idx in valid_cols:
                val = df.iloc[1, col_idx]
                if pd.notna(val) and str(val).strip():
                    t = str(val).strip().lower()
                    if t in ['number', 'num', 'numerical']:
                        prop_types.append('number')
                    else:
                        prop_types.append('text')
                else:
                    prop_types.append('text') # Empty treated as text

            # Map sequence headers strictly and exactly
            header_to_idx = {h: idx for idx, h in enumerate(viewer.full_headers)}
            node_updates = {}
            matched_count = 0
            unmatched_count = 0

            for df_row_idx in range(2, df.shape[0]):
                header_val = df.iloc[df_row_idx, 0]
                if pd.isna(header_val):
                    unmatched_count += 1
                    continue
                header_str = str(header_val).strip()
                
                # Enforce strict exact matching against viewer.full_headers
                if header_str in header_to_idx:
                    node_idx = header_to_idx[header_str]
                    node_updates[node_idx] = df_row_idx
                    matched_count += 1
                else:
                    unmatched_count += 1

            if matched_count == 0:
                raise ValueError("No matching sequence headers found. Enforced strict exact matching against full headers.")

            # Merge properties and types into memory
            for p_idx, prop_name in enumerate(prop_names):
                prop_type = prop_types[p_idx]
                col_idx = valid_cols[p_idx]

                # Check if property already exists
                if prop_name not in viewer.metadata:
                    # Initialize new property
                    if prop_type == 'number':
                        values = np.full(viewer.n_nodes, np.nan, dtype=np.float64)
                    else:
                        values = np.full(viewer.n_nodes, "", dtype=object)
                    viewer.metadata[prop_name] = {
                        "type": prop_type,
                        "values": values
                    }
                else:
                    # Overwrite type and safely convert existing values if type changes
                    old_type = viewer.metadata[prop_name]["type"]
                    viewer.metadata[prop_name]["type"] = prop_type
                    
                    if old_type != prop_type:
                        old_vals = viewer.metadata[prop_name]["values"]
                        if prop_type == 'number':
                            new_vals = np.full(viewer.n_nodes, np.nan, dtype=np.float64)
                            for i in range(viewer.n_nodes):
                                try:
                                    if str(old_vals[i]).strip():
                                        new_vals[i] = float(old_vals[i])
                                except ValueError:
                                    pass
                            viewer.metadata[prop_name]["values"] = new_vals
                        else:
                            new_vals = np.full(viewer.n_nodes, "", dtype=object)
                            for i in range(viewer.n_nodes):
                                if pd.notna(old_vals[i]):
                                    new_vals[i] = str(old_vals[i])
                            viewer.metadata[prop_name]["values"] = new_vals

                # Apply updates from file (ignore empty cells to avoid overwriting with blanks)
                values_arr = viewer.metadata[prop_name]["values"]
                for node_idx, df_row_idx in node_updates.items():
                    cell_val = df.iloc[df_row_idx, col_idx]
                    if pd.isna(cell_val) or str(cell_val).strip() == "" or str(cell_val).strip().lower() == "nan":
                        continue # Skip empty cells to preserve existing viewer metadata
                    
                    if prop_type == 'number':
                        try:
                            values_arr[node_idx] = float(cell_val)
                        except (ValueError, TypeError):
                            pass
                    else:
                        values_arr[node_idx] = str(cell_val)

            successful_files.append(filename)
            matched_nodes.update(node_updates.keys())
            total_unmatched += unmatched_count
            all_merged_props.update(prop_names)

        except Exception as e:
            failed_files.append((filename, str(e)))
            print(f"Error uploading metadata from {filename}: {e}")
            import traceback
            traceback.print_exc()

    # Cache metadata to the active .h5 layout file if any success
    if successful_files:
        cache_path, _ = utils.get_cache_filename()
        if os.path.exists(cache_path):
            try:
                with h5py.File(cache_path, "a") as hf:
                    if "metadata" in hf:
                        del hf["metadata"]
                    meta_group = hf.create_group("metadata")
                    for p_name, p_data in viewer.metadata.items():
                        p_type = p_data["type"]
                        vals = p_data["values"]
                        if p_type == "number":
                            ds = meta_group.create_dataset(p_name, data=vals, compression="gzip")
                        else:
                            dt_str = h5py.string_dtype(encoding='utf-8')
                            ds = meta_group.create_dataset(p_name, data=np.array(vals, dtype=object), dtype=dt_str, compression="gzip")
                        ds.attrs["type"] = p_type
            except Exception as e:
                print(f"Warning: Failed to write metadata to cache file: {e}")

    # Build response message
    msg_parts = []
    if successful_files:
        msg_parts.append(
            f"Successfully uploaded metadata from {len(successful_files)} file(s): {', '.join(successful_files)}. "
            f"Matched {len(matched_nodes)} unique nodes, ignored {total_unmatched} rows. "
            f"Merged properties: {', '.join(sorted(all_merged_props))}."
        )
    if failed_files:
        fail_details = "; ".join([f"{f}: {err}" for f, err in failed_files])
        msg_parts.append(f"Failed to upload from {len(failed_files)} file(s): {fail_details}")

    msg = " ".join(msg_parts)
    Command_Engine.print_help(viewer, msg)
