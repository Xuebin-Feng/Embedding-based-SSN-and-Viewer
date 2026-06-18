import os
import fnmatch
import SSN_Config as cfg
import Command_Engine
from PyQt6 import QtWidgets

def print_help():
    print("""
    Alignment Switcher Tool
    =======================
    Usage:
      alignment
          Opens a file explorer to select an MSA file (.fasta or .h5) manually.
      alignment <alignment_identifier>
          Loads the MSA file matching the identifier (index or filename query) 
          from the Multiple Alignments folder.
      alignment list
          Lists all available alignments in the Multiple Alignments folder.
      alignment help
          Displays this help message.

    Loading Rules:
      To load successfully, the FASTA subset currently representing the nodes in the viewer 
      must be a strict subset of the sequence headers present in the new MSA file. 
      If any node is missing, the load fails and the system automatically rolls back 
      to the previously active alignment to ensure session stability.
    """)

def run(viewer, args):
    # Check for help argument
    if args and args[0].lower() in ['help', '-h', '--help']:
        print_help()
        if hasattr(viewer, 'console_text'):
            viewer.console_text.text = "Help printed to console."
        return

    # Check for list argument
    msa_dir = getattr(cfg, 'MSA_DIR', os.path.join("Input_Files", "Multiple_Alignments"))
    
    # 1. No arguments provided -> Open File Explorer for manual selection
    if not args:
        try:
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                viewer.canvas.native,
                "Select Alignment File",
                msa_dir if os.path.exists(msa_dir) else "",
                "Alignment Files (*.fasta *.h5);;FASTA Files (*.fasta);;HDF5 Files (*.h5);;All Files (*)"
            )
        except Exception as e:
            msg = f"Error opening file dialog: {e}"
            Command_Engine.print_help(viewer, msg)
            return

        if not file_path:
            msg = "Alignment selection cancelled."
            Command_Engine.print_help(viewer, msg)
            return

        if not os.path.exists(file_path):
            msg = f"Error: File '{file_path}' does not exist."
            Command_Engine.print_help(viewer, msg)
            return

        selected_file = os.path.basename(file_path)
        new_path = file_path.replace("\\", "/")
    else:
        # Check if list is requested
        if args[0].lower() == 'list':
            if not os.path.exists(msa_dir):
                msg = f"Error: MSA directory '{msa_dir}' does not exist."
                Command_Engine.print_help(viewer, msg)
                return
            files = sorted([f for f in os.listdir(msa_dir) if f.endswith('.fasta') or f.endswith('.h5')])
            if not files:
                msg = f"No alignment files found in '{msa_dir}'."
                Command_Engine.print_help(viewer, msg)
                return
            print("\nAvailable alignments:")
            print("=====================")
            current_base = os.path.basename(cfg.MSA_FILE) if cfg.MSA_FILE else ""
            for i, file in enumerate(files, 1):
                is_active = "[ACTIVE]" if file == current_base else ""
                print(f"  {i: >2}. {file} {is_active}")
            print("\nTo load an alignment, type: alignment <index> or alignment <filename_query>")
            if hasattr(viewer, 'console_text'):
                viewer.console_text.text = f"Listed {len(files)} alignments in console."
            return

        # Treat args as target identifier in cfg.MSA_DIR
        if not os.path.exists(msa_dir):
            msg = f"Error: MSA directory '{msa_dir}' does not exist."
            Command_Engine.print_help(viewer, msg)
            return
            
        files = sorted([f for f in os.listdir(msa_dir) if f.endswith('.fasta') or f.endswith('.h5')])
        if not files:
            msg = f"Error: No alignment files found in '{msa_dir}'."
            Command_Engine.print_help(viewer, msg)
            return

        identifier = " ".join(args).strip()
        selected_file = None
        
        if identifier.isdigit():
            idx = int(identifier) - 1
            if 0 <= idx < len(files):
                selected_file = files[idx]
            else:
                msg = f"Error: Index '{identifier}' is out of range. Range is 1-{len(files)}."
                Command_Engine.print_help(viewer, msg)
                return
        else:
            # Search by case-insensitive substring
            matches = [f for f in files if identifier.lower() in f.lower()]
            if not matches:
                matches = [f for f in files if fnmatch.fnmatch(f.lower(), identifier.lower())]
                
            if len(matches) == 1:
                selected_file = matches[0]
            elif len(matches) > 1:
                print(f"\nMultiple matches found for '{identifier}':")
                for f in matches:
                    print(f"  - {f}")
                msg = "Error: Ambiguous query. Please be more specific."
                Command_Engine.print_help(viewer, msg)
                return
            else:
                msg = f"Error: No alignments matching '{identifier}' found."
                Command_Engine.print_help(viewer, msg)
                return

        new_path = os.path.join(msa_dir, selected_file).replace("\\", "/")

    # Load the selected alignment file
    print(f"\nAttempting to load alignment: {selected_file}...")
    if hasattr(viewer, 'console_text'):
        viewer.console_text.text = f"Loading {selected_file}..."

    # Backup current state for safety rollback
    backup_msa_file = cfg.MSA_FILE
    backup_alignment = viewer.alignment
    backup_active_ref = viewer.active_reference

    cfg.MSA_FILE = new_path

    try:
        viewer.load_global_alignment()

        # Check for missing reference sequence fallback
        if viewer.alignment is None or viewer.alignment.aln is None:
            print(f"Warning: Active reference '{viewer.active_reference}' not found in '{selected_file}'.")
            print("Re-attempting load in Pure Occupancy Mode (no reference sequence)...")
            viewer.active_reference = None
            viewer.load_global_alignment()

            if viewer.alignment is None or viewer.alignment.aln is None:
                raise ValueError("Alignment loader failed to return an alignment.")
            else:
                success_msg = f"Success: Loaded '{selected_file}' in Pure Occupancy Mode."
                print(f"\n{success_msg}")
                if hasattr(viewer, 'console_text'):
                    viewer.console_text.text = f"Loaded {selected_file} (No Ref)"
        else:
            success_msg = f"Success: Loaded alignment '{selected_file}'."
            print(f"\n{success_msg}")
            if hasattr(viewer, 'console_text'):
                viewer.console_text.text = f"Loaded {selected_file}"

    except Exception as e:
        # Check if it was a subset violation error
        err_msg = str(e)
        if "subset" in err_msg.lower() or "missing" in err_msg.lower():
            explanation = (
                "\nCRITICAL ERROR: Sequence subset violation!\n"
                "The active network sequence set (FASTA subset) must be a strict subset of the sequences in the MSA.\n"
                "One or more nodes in the current view do not exist in the new alignment."
            )
            print(explanation)
        
        print(f"\nFailed to load alignment '{selected_file}': {e}")
        print("Reverting to previous alignment state...")
        
        # Rollback
        cfg.MSA_FILE = backup_msa_file
        viewer.alignment = backup_alignment
        viewer.active_reference = backup_active_ref
        
        if hasattr(viewer, 'console_text'):
            viewer.console_text.text = "Load failed. Reverted to previous alignment."
