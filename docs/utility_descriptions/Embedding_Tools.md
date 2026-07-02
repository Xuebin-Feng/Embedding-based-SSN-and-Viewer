# 🧬 Embedding Injection (`Embedding_Injection.py`)

This script injects new sequence embeddings into an existing HDF5 embedding database. It scans the incoming FASTA sequence list, extracts pre-computed embeddings directly from the database for existing matches, and computes embeddings only for the newly added sequences to optimize compute time.

### 📥 Input

#### Target Embedding Database `INPUT_EMBED`
*   **Format**: Existing master HDF5 embedding database (`.h5`).
*   **Created By**: `Generate_Embeddings.py` (Embedding Generation utility).

#### Incoming Sequence Set `INPUT_FASTA`
*   **Format**: FASTA file (`.fasta`) containing original sequences plus newly appended targets.
*   **Created By**: User-compiled updated sequence set.

### ⚙️ Parameters

This script does not require additional configuration parameters.

### 📤 Output

#### Updated HDF5 Embedding Database
*   **Format**: HDF5 (`.h5`).
*   **Description**: Re-indexed database file containing embeddings for all sequences in the new FASTA file.

<details markdown="1">
<summary><b>Algorithm Details</b></summary>

1. **Header Inventory and Discrepancy Parsing**:
     Reads the new input FASTA file and collects all target headers:
     $$H_{\text{fasta}} = \{h_1, h_2, \dots, h_M\}$$
     
     Opens the existing HDF5 embedding file and reads the pre-computed headers:
     $$H_{\text{exist}} = \{e_1, e_2, \dots, e_N\}$$
     
     Identifies the subset of new sequences to embed using set subtraction:
     $$H_{\text{new}} = H_{\text{fasta}} \setminus H_{\text{exist}}$$

2. **Model Identification and Setup**:
     Reads the metadata of the existing HDF5 file to identify the model architecture and precision (`float16` or `float32`) used. It loads the exact same model (e.g. ESM-C) to ensure vector consistency.

3. **Incremental Embedding Computation**:
     Feeds the sequence segments belonging to **H<sub>new</sub>** through the language model, calculating their residue-level embeddings.

4. **Synchronized Merge and HDF5 Serialization**:
     Iterates through $H_{\text{fasta}}$ in order. If a header belongs to $H_{\text{exist}}$, it copies the embedding dataset directly from the old file. If it belongs to $H_{\text{new}}$, it writes the newly computed embedding tensor:
     $$v_{\text{final}}(i) = \begin{cases} v_{\text{exist}}(i) & \text{if } h_i \in H_{\text{exist}} \\ v_{\text{new}}(i) & \text{otherwise} \end{cases}$$
     
     Saves the re-indexed embedding datasets, headers, and sequence lengths.

</details>

---

# 📤 Embedding Extraction (`Embedding_Extraction.py`)

This script extracts a subset of sequence embeddings from a master HDF5 database. By providing a list of target sequence headers (either as a FASTA or text file), it creates a smaller, filtered HDF5 embedding archive without running any model calculations.

### 📥 Input

#### Source Embedding Database `INPUT_EMBED`
*   **Format**: Master HDF5 embedding database file (`.h5`).
*   **Created By**: `Generate_Embeddings.py` (Embedding Generation utility).

#### Target Whitelist Set `INPUT_FASTA`
*   **Format**: Target whitelist FASTA file (`.fasta`) or plain text file containing selected headers.
*   **Created By**: User-defined subset whitelist.

### ⚙️ Parameters

This script does not require additional configuration parameters.

### 📤 Output

#### Extracted HDF5 Embedding Archive
*   **Format**: HDF5 (`.h5`).
*   **Description**: Contains whitelisted embeddings copied directly from the master database.

<details markdown="1">
<summary><b>Algorithm Details</b></summary>

1. **Target List Gathering**:
     Parses the target sequence whitelist (from a FASTA file or text list) to compile the target headers:
     $$H_{\text{target}} = \{t_1, t_2, \dots, t_K\}$$

2. **Index Alignment and Intersection**:
     Iterates through the master HDF5 file's header dataset and filters out any sequence datasets not present in $H_{\text{target}}$:
     $$H_{\text{extract}} = H_{\text{target}} \cap H_{\text{master}}$$

3. **Batch dataset extraction**:
     Copies the multidimensional embedding datasets corresponding to **H<sub>extract</sub>** directly from the master HDF5 file without decompression or modifications.

4. **Metadata Serialization**:
     Writes the extracted datasets, updated headers array, and sequence lengths to the new target HDF5 output file.

</details>
