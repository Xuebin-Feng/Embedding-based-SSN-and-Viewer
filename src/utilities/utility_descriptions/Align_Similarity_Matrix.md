# 🧬 Dynamic Programming Embedding Alignment (`Align_Similarity_Matrix.py`)

This script computes an all-vs-all sequence similarity network (SSN) using residue-level protein embeddings. Instead of using traditional amino acid substitution matrices, it calculates similarity matrices by comparing the dense high-dimensional embedding vectors of each residue.

It performs sequence alignment scoring using dynamic programming (Smith-Waterman for local alignment and Needleman-Wunsch for global alignment) implemented in optimized Numba JIT functions for maximum performance. To accelerate alignment of massive datasets, it includes an optional pre-filtering step that computes the global cosine similarity of sequence mean embeddings to skip full alignments for highly dissimilar sequences.

### 📥 Input

#### HDF5 Embedding Database `INPUT_HDF5`
*   **Format**: HDF5 (`.h5`).
*   **Created By**: `Generate_Embeddings.py` (Embedding Generation utility).
*   **Description**: Contains pre-calculated residue embeddings as $L \times D$ arrays.

### ⚙️ Parameters

| Parameter | Description |
| :--- | :--- |
| Enable Edge Pre-filtering **`EDGE_PREFILTERING`** | Toggle to enable pre-filtering sequence pairs using global cosine similarity. If enabled, full residue-level alignments are skipped for pairs below the similarity cutoff. |
| Pre-filtering Strength **`PREFILTER_STRENGTH`** | The percentage of sequence pairs with the lowest global cosine similarity scores to exclude from alignment calculations. Higher values increase speed but risk missing alignments between distantly related proteins. |
| Generate Traceback Paths **`GENERATE_PATHS`** | Toggle to calculate and save the traceback alignment paths (match/gap index mappings) in a separate paths file. If OFF, saves significant disk space. |
| CPU Worker Threads **`WORKERS`** | The number of CPU processes/threads allocated for parallel alignment calculations. |
| Local Gap Penalty **`LOCAL_GAP_P`** | The gap penalty score applied for local alignments. More negative values penalize gaps more heavily, resulting in fewer gaps. |
| Global Gap Penalty **`GLOBAL_GAP_P`** | The gap penalty score applied for global alignments. |
| Processing Batch Size **`BATCH_SIZE`** | The number of sequence pairs processed in a single chunk before writing to disk. Larger batches maximize CPU core utilization but increase memory consumption. Can be set to a number or 'auto'. |

### 📤 Output

#### HDF5 Alignment Network
*   **Format**: HDF5 (`.h5`).
*   **Structure**:
    - `/i`: Source sequence node indices.
    - `/j`: Target sequence node indices.
    - `/g_score`: Global alignment scores.
    - `/g_len`: Global alignment lengths.
    - `/l_score`: Local alignment scores.
    - `/l_len`: Local alignment lengths.
    - `/headers`: Array of sequence headers.

#### HDF5 Alignment Paths File (Optional)
*   **Format**: HDF5 (`.h5`).
*   **Created When**: `GENERATE_PATHS` parameter is set to True.
*   **Structure**: Contains zlib-compressed packed traceback path arrays mapped to sequence indices. Saves matching/gap alignments for pairwise visualization.

<details markdown="1">
<summary><b>Algorithm Details</b></summary>

The alignment pipeline is executed in the following steps:

1. **Global Embedding Pooling**:
     For each protein sequence, its residue embeddings are pooled (either via mean or max pooling) to generate a single global sequence representation vector:
     $$u_i = \text{mean}_{\text{residues}}(\text{emb}_i(\text{residue}))$$

2. **Edge Pre-filtering (Optional)**:
     If pre-filtering is enabled, all-vs-all cosine similarities are calculated on the CPU:
     $$\text{Sim}_{\text{cos}}(i, j) = \frac{u_i \cdot u_j}{\|u_i\|_2 \times \|u_j\|_2}$$
     
     This similarity is adjusted by the sequence length ratio:
     $$\text{Adj}(i, j) = \text{Sim}_{\text{cos}}(i, j) \times \left(\frac{\min(L_i, L_j)}{\max(L_i, L_j)}\right)^P$$
     
     Pairs scoring in the bottom percentile corresponding to `PREFILTER_STRENGTH` are skipped.

3. **Residue-Level Similarity Matrix Computation**:
     For each sequence pair $(i, j)$ passing the pre-filter, a pairwise Cosine distance matrix is computed. First, the residue embeddings are L2-normalized along the hidden dimension:
     $$\hat{v}_i(a) = \frac{v_i(a)}{\|v_i(a)\|_2}$$
     
     Then, the cosine distance is calculated using matrix multiplication:
     $$D(a, b) = 1.0 - \hat{v}_i(a) \cdot \hat{v}_j(b)$$
     
     where $v_i(a)$ is the embedding vector for residue $a$ in sequence $i$. This distance is converted into a similarity matrix:
     $$S(a, b) = \exp(-D(a, b))$$

4. **Dual Z-Score Normalization**:
     The similarity matrix is normalized row-wise and column-wise to adjust for residue-specific background similarities:
     $$Z_{\text{row}}(a, b) = \frac{S(a, b) - \mu_{\text{row}}(a)}{\sigma_{\text{row}}(a) + \varepsilon}$$
     $$Z_{\text{col}}(a, b) = \frac{S(a, b) - \mu_{\text{col}}(b)}{\sigma_{\text{col}}(b) + \varepsilon}$$
     
     The final alignment scoring matrix is the average of these Z-scores:
     $$\text{Score}(a, b) = \frac{Z_{\text{row}}(a, b) + Z_{\text{col}}(a, b)}{2}$$

5. **Dynamic Programming Alignment**:
     * **Needleman-Wunsch (Global Pass)**: Solves the standard global recurrence relation using the scoring matrix $\text{Score}(a, b)$ and `GLOBAL_GAP_P` to output a global score.
     * **Smith-Waterman (Local Pass)**: Subtracts a shift value of 2.0 from the scoring matrix (to ensure dissimilar matches have negative scores):
       $$\text{Score}_{\text{local}}(a, b) = \text{Score}(a, b) - 2.0$$
       
       It then runs the standard local dynamic programming recurrence with `LOCAL_GAP_P` to identify the optimal local alignment score.

6. **Serialization & Path Packing**:
     If `GENERATE_PATHS` is enabled, the global alignment traceback directions are encoded as a 2-bit packed array, compressed using zlib, and saved to the output network.

</details>
