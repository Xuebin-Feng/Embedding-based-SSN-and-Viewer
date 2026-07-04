import sys
import os
import json
import shutil

# Add workspace to sys.path
workspace = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(workspace)

import commands.assist as assist

# Expected test dataset (5 use-cases per command, ranging from casual to professional)
TEST_CASES = [
    # 1. alignment
    ("open alignment file picker", "alignment"),
    ("select alignment file manually", "alignment"),
    ("load alignment from Input_Files/Multiple_Alignments/SeqSet_alignment.fasta", "alignment Input_Files/Multiple_Alignments/SeqSet_alignment.fasta"),
    ("use alignment file SeqSet_alignment.fasta", "alignment SeqSet_alignment.fasta"),
    ("load alignment SeqSet_alignment", "alignment SeqSet_alignment"),
    
    # 2. cluster
    ("cluster the SSN", "cluster"),
    ("run clustering on the network", "cluster"),
    ("cluster with parameter 1.2 and minimum cluster size 20", "cluster leiden 1.2 20"),
    ("run mcl clustering with inflation 2.5 and min size 15", "cluster mcl 2.5 15"),
    ("list all generated clusters", "cluster list"),
    
    # 3. color
    ("make the selected nodes red", "color $sele$ red"),
    ("color cluster 2 green", "color #cluster_2# green"),
    ("double the size of nodes in cluster 3", "color #cluster_3# x2"),
    ("make nodes with length > 500 yellow circles", "color {Length>500} yellow circle"),
    ("color nodes with alanine at position 50 orange", "color A50 orange"),
    
    # 4. export
    ("export clusters", "export"),
    ("save all sequence clusters as fasta", "export"),
    ("export the human custom group", "export group:human"),
    ("save human group sequences as fasta", "export group:human"),
    ("export all defined custom groups to fasta files", "export group"),
    
    # 5. group
    ("group selected as kinase", "group kinase"),
    ("label selection as active_site", "group active_site"),
    ("add cluster 1 nodes to group receptor", "group #cluster_1# receptor"),
    ("list all custom groups", "group list"),
    ("delete active_site group", "group remove active_site"),
    
    # 6. hide
    ("hide selected", "hide"),
    ("hide cluster 2", "hide #cluster_2#"),
    ("hide singletons", "hide single"),
    ("hide nodes with no edges", "hide single"),
    ("hide nodes with length less than 300", "hide {Length<300}"),
    
    # 7. label
    ("run subset analysis", "label"),
    ("compare cluster residues", "label"),
    ("compare custom groups setting cmin to 90%", "label groups cmin 90%"),
    ("run residue analysis on custom groups setting gmax to 40%", "label groups gmax 40%"),
    ("perform differential labeling comparison with cmin 95%", "label cmin 95%"),
    
    # 8. logo
    ("make logo for position 10 to 20", "logo [10-20]"),
    ("sequence logo for cluster 1 from position 50 to 60", "logo #cluster_1# [50-60]"),
    ("generate percentage logo for positions 5 and 6 without gaps", "logo [5,6] pcts no_gap"),
    ("sequence logo for selected nodes at positions 12, 15, and 18", "logo $sele$ [12,15,18]"),
    ("create a sequence logo for cluster 3 at range 100-110 in percentages", "logo #cluster_3# [100-110] pcts"),
    
    # 9. meta
    ("load metadata", "meta"),
    ("upload metadata sheet metadata.xlsx", "meta metadata.xlsx"),
    ("show Organism property when clicked", "meta show Organism"),
    ("clear HUD display", "meta show clear"),
    ("export metadata for cluster 1 to excel", "meta download #cluster_1#"),
    
    # 10. print
    ("take a snapshot", "print"),
    ("save screenshot", "print"),
    ("take transparent background snapshot", "print transparent"),
    ("save a vector graphic of the ssn", "print svg"),
    ("stitch high resolution transparent snapshot of the network", "print full transparent"),
    
    # 11. query
    ("query position 106", "query [106]"),
    ("check residue distribution at column 50", "query [50]"),
    ("query alignment columns 10 to 15", "query [10-15]"),
    ("check consensus residues for columns 5, 8, and 12", "query [5,8,12]"),
    ("query active alignment column 45", "query [45]"),
    
    # 12. redo
    ("redo", "redo"),
    ("redo change", "redo"),
    ("reapply last undone action", "redo"),
    ("redo last action", "redo"),
    ("reapply the undone change", "redo"),
    
    # 13. reference
    ("show reference", "reference"),
    ("what is the current reference sequence", "reference"),
    ("set reference to SeqA", "reference SeqA"),
    ("change reference sequence to SeqB", "reference SeqB"),
    ("switch global alignment reference to SeqC", "reference SeqC"),
    
    # 14. reset
    ("reset color", "reset colors"),
    ("reset sizes and layout", "reset sizes\nreset network"),
    ("unhide all nodes", "reset hide"),
    ("clear all clusters", "reset clusters"),
    ("reset all shapes to default", "reset shapes"),
    
    # 15. save
    ("save network", "save"),
    ("save layout", "save"),
    ("save layout state as my_layout.h5", "save my_layout.h5"),
    ("save the layout state", "save"),
    ("persist current network session as ssn_backup.h5", "save ssn_backup.h5"),
    
    # 16. select
    ("select cluster 1", "select #cluster_1#"),
    ("add nodes containing 3HMU to selection", "select add \"3HMU\""),
    ("remove noise nodes from selection", "select remove #noise#"),
    ("keep only nodes with length >= 400", "select keep {Length>=400}"),
    ("invert selection", "select invert"),
    
    # 17. spectrum
    ("color by length", "spectrum prop:Length"),
    ("color nodes by length using magma scheme", "spectrum prop:Length scheme:magma"),
    ("apply viridis color gradient based on property Weight", "spectrum prop:Weight scheme:viridis"),
    ("color cluster 1 by sequence length", "spectrum #cluster_1# prop:Length"),
    ("color nodes in organism coli by length using coolwarm scheme", "spectrum {Organism=*coli*} prop:Length scheme:coolwarm"),
    
    # 18. subcluster
    ("subcluster cluster 1", "subcluster cluster_1"),
    ("run subclustering on cluster 1", "subcluster cluster_1"),
    ("subcluster cluster 2 using leiden with parameter 1.5", "subcluster cluster_2 leiden 1.5"),
    ("subcluster cluster 3 using mcl with inflation 3.0", "subcluster cluster_3 mcl 3.0"),
    ("run subclustering on cluster 4 with minimum size 5", "subcluster cluster_4 5"),
    
    # 19. undo
    ("undo", "undo"),
    ("undo change", "undo"),
    ("revert last action", "undo"),
    ("undo previous command", "undo"),
    ("revert the previous layout state", "undo"),
    
    # 20. zoom
    ("zoom to 500", "zoom 500"),
    ("zoom camera to 400 units", "zoom 400"),
    ("snap view width to 600", "zoom 600"),
    ("zoom camera view width to 350", "zoom 350"),
    ("set camera view range to 800", "zoom 800"),
    
    # Compound / Multi-command cases
    ("reset the colors first, and then color all nodes with aspartate at position 25 to red", "reset colors\ncolor D25 red"),
    ("select nodes with length > 500 and zoom to 600", "select {Length>500}\nzoom 600"),
    ("load metadata file metadata.xlsx and color D1_len using spectrum", "meta metadata.xlsx\nspectrum prop:D1_len"),
    ("hide all singletons and reset layout", "hide single\nreset network")
]

class MockViewer:
    def __init__(self):
        self.llm_loaded = False
        self.llm_backend = None
        self.llm_url = None
        self.llm_model_name = None
        self.llm_temperature = 0.0

def main():
    print("====================================================")
    print("Starting LLM CLI Agent Evaluation & Self-Correction")
    print("====================================================")
    
    # Initialize mock viewer and load configuration
    viewer = MockViewer()
    success = assist.activate_assistant(viewer, quiet=True)
    if not success or not viewer.llm_loaded:
        print("Error: Could not load any active LLM backend from assist_config.json.")
        print("Please start Ollama/LM Studio or add a GGUF model in commands/assist_resource/.")
        sys.exit(1)
        
    print(f"Connected to backend: {viewer.llm_backend} (model: {viewer.llm_model_name})")
    
    # Read system prompt
    prompt_path = os.path.join(workspace, "commands", "assist_resource", "system_prompt.md")
    backup_path = prompt_path + ".bak"
    
    # Backup prompt file
    shutil.copyfile(prompt_path, backup_path)
    print(f"Backed up system prompt to {os.path.basename(backup_path)}")
    
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except Exception as e:
        print(f"Error reading system prompt: {e}")
        sys.exit(1)
        
    failures = []
    successes = 0
    
    for i, (query, expected) in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}] Testing query: '{query}'")
        print(f"  Expected:\n{expected}")
        
        # Call active backend
        translated = None
        try:
            if viewer.llm_backend == "server":
                translated = assist.call_api(viewer.llm_url, viewer.llm_model_name, system_prompt, query, history=None, temperature=viewer.llm_temperature)
            elif viewer.llm_backend == "gguf":
                translated = assist.call_gguf(viewer.llm_model, system_prompt, query, history=None, temperature=viewer.llm_temperature)
        except Exception as e:
            print(f"  Error calling LLM: {e}")
            failures.append((query, expected, f"LLM Call Error: {e}"))
            continue
            
        if not translated:
            print("  Error: No response from LLM.")
            failures.append((query, expected, "No response from LLM"))
            continue
            
        # Parse translated command (strip codeblocks)
        cmd_lines = []
        in_code_block = False
        for line in translated.split("\n"):
            line_strip = line.strip()
            if line_strip.startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                if line_strip: cmd_lines.append(line_strip)
            else:
                if line_strip and not line_strip.lower().startswith("input:") and not line_strip.lower().startswith("output:"):
                    clean_line = line_strip.strip("`'")
                    if clean_line: cmd_lines.append(clean_line)
                    
        translated_clean = "\n".join(cmd_lines).strip()
        print(f"  Got:\n{translated_clean}")
        
        # Compare strings exactly (ignoring differences in line-breaks whitespace)
        expected_norm = "\n".join([line.strip() for line in expected.split("\n") if line.strip()]).lower()
        translated_norm = "\n".join([line.strip() for line in translated_clean.split("\n") if line.strip()]).lower()
        
        if expected_norm == translated_norm:
            print("  Result: SUCCESS")
            successes += 1
        else:
            print("  Result: FAIL (Mistake detected)")
            failures.append((query, expected, translated_clean))
            
    print("\n=================== EVALUATION RESULTS ===================")
    print(f"Total Tests: {len(TEST_CASES)}")
    print(f"Successes:   {successes}")
    print(f"Failures:    {len(failures)}")
    print("==========================================================")
    
    # Self-Correction Step
    if failures:
        print("\nApplying self-correction updates to system_prompt.md...")
        updated_prompt = system_prompt
        
        appended_count = 0
        for query, expected, got in failures:
            # Check if this exact example is already in the system prompt to prevent duplicate appends
            example_pattern = f"Input: {query}"
            if example_pattern not in updated_prompt:
                # Format the failed example as a few-shot addition
                new_example = f"\n\nInput: {query}\nOutput:\n{expected}"
                updated_prompt += new_example
                appended_count += 1
                print(f"  Added correct example for: '{query}'")
                
        if appended_count > 0:
            try:
                with open(prompt_path, "w", encoding="utf-8") as f:
                    f.write(updated_prompt)
                print(f"Successfully appended {appended_count} correction examples to system_prompt.md.")
            except Exception as e:
                print(f"Error saving system prompt corrections: {e}")
        else:
            print("No new correction examples needed (failed cases were already present in prompt).")
            
    else:
        print("\nAll translations matched perfectly! No prompt updates required.")

if __name__ == "__main__":
    main()
