# 🧬 Network Injection (`Network_Injection.py`)

This script performs incremental similarity network calculations. When new sequences are added to a project, it copies all pre-existing sequence-to-sequence alignment scores directly from the old network cache, aligning only the newly introduced sequence pairs to save time and compute resources.

### 📥 Input

#### Target Network File `OLD_NETWORK`
*   **Format**: Pre-existing HDF5 network database file (`.h5`).
*   **Created By**: `Align_Similarity_Matrix.py` (Embedding Alignment utility).

#### Updated Embedding Database `NEW_EMBEDDINGS`
*   **Format**: Target HDF5 embedding database (`.h5`) containing all embeddings.
*   **Created By**: `Embedding_Injection.py` (Embedding Injection utility).

### ⚙️ Parameters

| Parameter | Description |
| :--- | :--- |
| Local Gap Penalty **`LOCAL_GAP_P`** | The gap open/extend penalty applied during local Smith-Waterman alignment calculations. |
| Global Gap Penalty **`GLOBAL_GAP_P`** | The gap open/extend penalty applied during global Needleman-Wunsch alignment calculations. |
| CPU Worker Threads **`WORKERS`** | The number of CPU threads allocated for parallel alignment of new sequence pairs. |
| Processing Batch Size **`BATCH_SIZE`** | The number of sequence pairs aligned in a single multiprocessing block before writing results to disk, minimizing VRAM and RAM footprint. |

### 📤 Output

#### Updated HDF5 Alignment Network
*   **Format**: HDF5 (`.h5`).
*   **Description**: Output network combining pre-existing edge scores with the newly aligned pairs.

<details markdown="1">
<summary><b>Algorithm Details</b></summary>

1. **Mapping Setup**:
     Let the old network headers be $H_{\text{old}} = \{h_1, \dots, h_N\}$ and the new embedding headers be $H_{\text{new}} = \{h'_1, \dots, h'_M\}$ (where $M > N$ and $H_{\text{old}} \subseteq H_{\text{new}}$).
     The script creates an index mapping dictionary to resolve old indices to new indices:
     $$\text{Map}_{\text{old} \to \text{new}}(i) = j \quad \text{such that} \quad h_i = h'_j$$

2. **Edge Classification**:
     For all pairwise combinations in the new network (u, v) (where 0 ≤ u < v < M):
     - **Case 1 (Pre-existing Pair)**: If both *u* and *v* exist in **H<sub>old</sub>**, the edge score is copied directly from the old network cache.
     - **Case 2 (New Pair)**: If either *u* or *v* (or both) do not exist in **H<sub>old</sub>**, the pair is scheduled for active dynamic programming alignment.

3. **Incremental Multiprocessed Alignment**:
     Sends the scheduled new pairs to parallel CPU workers. Each worker:
     - Retrieves residue embeddings from the HDF5 database.
     - Calculates the normalized score matrix:
       $$\text{Score}(a, b) = \frac{Z_{\text{row}}(a, b) + Z_{\text{col}}(a, b)}{2}$$
     - Solves global and local dynamic programming alignments:
       $$\text{Global Pass} \to \text{NW}(\text{Score}, \text{gap}_g)$$
       $$\text{Local Pass} \to \text{SW}(\text{Score} - 2.0, \text{gap}_l)$$

4. **Consolidation**:
     Merges the copied scores with the newly calculated scores and writes the updated re-indexed network datasets (`i`, `j`, `g_score`, `g_len`, `l_score`, `l_len`, `path`) to the new output file.

</details>

---

# 📤 Network Extraction (`Network_Extraction.py`)

This script extracts sub-networks from a master HDF5 network based on a whitelist FASTA file. It retains only the alignment connections where both sequence nodes are in the whitelist, and re-indexes all remaining edge indices to produce a clean, self-contained filtered sub-network.

### 📥 Input

#### Source Network File `INPUT_NET`
*   **Format**: Master HDF5 network database file (`.h5`).
*   **Created By**: `Align_Similarity_Matrix.py` (Embedding Alignment utility) or `Align_Substitution_Matrix.py` / `Parse_BLAST_Output.py`.

#### Target Whitelist Set `INPUT_FASTA`
*   **Format**: Whitelist sequence FASTA file (`.fasta`) containing nodes to retain.
*   **Created By**: User-defined subset whitelist.

### ⚙️ Parameters

This script does not require additional configuration parameters.

### 📤 Output

#### Extracted HDF5 Sub-Network Archive
*   **Format**: HDF5 (`.h5`).
*   **Description**: Contains whitelisted network edges (`i`, `j`, `g_score`, `l_score`, `headers`) re-indexed to match the subset.

<details markdown="1">
<summary><b>Algorithm Details</b></summary>

1. **Whitelist Indexing**:
     Loads the target headers whitelist $H_{\text{whitelist}}$ from the FASTA file. Maps each whitelist header to its corresponding index in the master network file:
     $$\text{Map}_{\text{header} \to \text{master\_idx}}(h) = x$$
     
     It then establishes a new index mapping for the subset:
     $$\text{Map}_{\text{master\_idx} \to \text{subset\_idx}}(x) = y$$

2. **Edge Filtering**:
     Scans the master network edges $(i_k, j_k)$. An edge is retained if and only if both indices are in the whitelist:
     $$i_k \in \text{Map}_{\text{master\_idx} \to \text{subset\_idx}} \quad \text{and} \quad j_k \in \text{Map}_{\text{master\_idx} \to \text{subset\_idx}}$$

3. **Re-Indexing**:
     For all retained edges, the script re-indexes the source and target node values to fit the smaller subset matrix coordinate space:
     $$i'_k = \text{Map}_{\text{master\_idx} \to \text{subset\_idx}}(i_k)$$
     $$j'_k = \text{Map}_{\text{master\_idx} \to \text{subset\_idx}}(j_k)$$

4. **Output Assembly**:
     Writes the filtered, re-indexed edges and their alignment scores (plus zlib-compressed traceback paths if available) to a compact standalone HDF5 network file.

</details>
