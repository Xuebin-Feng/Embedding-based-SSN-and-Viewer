# 🧬 Substitution Matrix Alignment (`Align_Substitution_Matrix.py`)

This script runs all-vs-all local sequence alignments using traditional amino acid substitution matrices. It constructs a local NCBI BLAST database from the input sequence set and executes parallelized BLASTP queries. The resulting E-values are converted into linearly comparable negative Log10(E) edge weights for network mapping.

### 📥 Input

#### Sequence FASTA File `INPUT_FASTA`
*   **Format**: Standard FASTA (`.fasta`).
*   **Created By**: `Sanitize_Sequences.py` (Sequence Sanitization utility) or user-provided raw FASTA.
*   **Description**: Raw sequence database to run BLAST against.

### ⚙️ Parameters

| Parameter | Description |
| :--- | :--- |
| Substitution Matrix **`MATRIX`** | The amino acid substitution matrix used for traditional scoring during BLAST alignments (e.g., BLOSUM45, BLOSUM50, BLOSUM62, BLOSUM80, BLOSUM90, PAM30, PAM70, PAM250). |
| BLAST Threads **`NUM_THREADS`** | The number of CPU threads allocated for BLASTP execution and parsing. |
| Processing Batch Size **`BATCH_SIZE`** | The number of parsed alignments accumulated in memory before writing to disk, protecting system RAM. |
| Temporary Working Directory **`SAFE_TEMP_DIR`** | The temporary directory used to cache intermediate query segments and BLAST output files during the parallel run. |

### 📤 Output

#### HDF5 Alignment Network
*   **Format**: HDF5 (`.h5`).
*   **Structure**:
    - `/i`: Source sequence node indices.
    - `/j`: Target sequence node indices.
    - `/l_score`: Calculated $-\log_{10}(E_{\text{value}})$ scores.
    - `/headers`: Array of sequence headers.

<details markdown="1">
<summary><b>Algorithm Details</b></summary>

1. **Header Sanitization**:
     To avoid parser failures in NCBI BLAST due to complex characters, spaces, or vertical bars (`|`), the script generates a mapping:
     $$\text{Header} \to \text{Index} \quad (0, 1, 2, \dots)$$
     
     It creates a temporary `safe_fasta` containing sequence headers renamed as their respective integer indices.

2. **Local BLAST Database Construction**:
     Executes `makeblastdb` to generate a local database using the sanitized FASTA.

3. **Multithreaded Query Chunking**:
     Splits the query FASTA into parallelized temporary chunks to distribute search workloads across the selected number of CPU workers (`NUM_THREADS`).

4. **Pairwise BLASTP Execution**:
     Executes `blastp` for each query chunk against the database:
     * **blastp** → tabular output format 6

5. **E-Value Conversion**:
     Extracts E-value scores for each hit and linearizes them into edge connectivity weights using a negative base-10 logarithm:
     $$\text{Score} = -\log_{10}(E_{\text{value}} + 10^{-300})$$
     
     The lower bound offset of $10^{-300}$ is added to avoid mathematical division by zero when the E-value is 0.0.

6. **Consolidation**:
     Merges chunked output files, resolves temporary indices back to original headers, and exports the final network file.

</details>

---

# 🔍 Parse BLAST Output (`Parse_BLAST_Output.py`)

This script parses pre-computed, tab-separated tabular BLAST outfmt 6 output files and converts them into standard HDF5 network files. It automatically detects which columns contain sequence headers and E-value variables, performs E-value conversion, and filters out redundant alignment edges to build clean networks.

### 📥 Input

#### Tabular BLAST Output File `INPUT_BLAST_TABULAR`
*   **Format**: Tab-separated tabular values (`.txt`, `.tab`, `.tsv`).
*   **Created By**: Externally run NCBI BLASTP command (`blastp -outfmt 6`).
*   **Structure**: Output columns from `blastp -outfmt 6`. Must contain query ID, subject ID, and E-value columns.

### ⚙️ Parameters

This script does not require additional configuration parameters.

### 📤 Output

#### HDF5 Alignment Network
*   **Format**: HDF5 (`.h5`).
*   **Structure**:
    - `/i`: Source sequence node indices.
    - `/j`: Target sequence node indices.
    - `/l_score`: Parsed $-\log_{10}(E_{\text{value}})$ scores.
    - `/headers`: Array of sequence headers.

<details markdown="1">
<summary><b>Algorithm Details</b></summary>

1. **Automatic Column Detection**:
     Scans the first 1000 lines of the tabular document using regular expressions to detect the column index $c_{\text{evalue}}$ that contains E-values (identifying scientific notations or exact 0.0 matches).

2. **Header Index Resolution**:
     Constructs a header-to-index mapping dictionary dynamically as it reads lines.

3. **Edge Parsing and Score Conversion**:
     Extracts query ID, subject ID, and E-value, and converts the E-value:
     $$\text{Score} = -\log_{10}(E_{\text{value}} + 10^{-300})$$

4. **Deduplication**:
     Retains only the highest-scoring alignment edge between any undirected sequence pair (u, v) to filter out redundant alignments.

5. **HDF5 Serialization**:
     Writes headers, sequence indices, and edge score datasets to the output network.

</details>
