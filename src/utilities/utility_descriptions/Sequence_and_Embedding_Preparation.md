# 🧼 Sanitize Sequences (`Sanitize_Sequences.py`)

This script cleans raw FASTA sequence databases to prepare them for language model embedding. It filters out sequences outside target size boundaries, excludes sequences containing specific header keywords (such as fragments or partials), replaces unsafe non-standard characters, and reports sequence length distributions.

### 📥 Input

#### Raw Sequence FASTA File `INPUT_FASTA`
*   **Format**: Standard FASTA format (`.fasta`, `.fa`, `.txt`).
*   **Created By**: User-provided raw protein sequence database.
*   **Structure**:
    ```text
    >Sequence_Header_1 [Optional Description]
    MNSGVSRRQ...
    >Sequence_Header_2
    MKVLLVSDA...
    ```

### ⚙️ Parameters

| Parameter | Description |
| :--- | :--- |
| Overwrite **`OVER_WRITE`** | Toggle to overwrite the input FASTA file with the cleaned sequences. If disabled, saves to a new file with a `_clean.fasta` suffix. |
| Enable Length Filter **`ENABLE_LENGTH_FILTER`** | Toggle to filter out sequences that do not meet the minimum or maximum length constraints. |
| Minimum Sequence Length **`MIN_SEQ_LENGTH`** | The minimum sequence length (in amino acids) required to keep a sequence. |
| Maximum Sequence Length **`MAX_SEQ_LENGTH`** | The maximum sequence length allowed. |
| Remove by Header String **`REMOVE_BY_HEADER_STRING`** | Excludes sequences whose headers contain this specific case-insensitive substring (e.g., 'partial', 'fragment', 'low quality'). Set to 'None' to disable. |

### 📤 Output

#### Sanitized FASTA File
*   **Format**: Standard FASTA (`.fasta`).
*   **Description**: Contains cleaned sequences where headers are sanitized (special characters replaced with underscores) and invalid characters are removed.

<details markdown="1">
<summary><b>Algorithm Details</b></summary>

1. **Header Sanitization**:
     Cleans header strings to make them safe for downstream pipeline tools. It replaces slashes, backslashes, colons, semi-colons, brackets, parentheses, spaces, and commas with underscores:
     $$h_{\text{clean}} = \text{replace\_unsafe}(h_{\text{raw}})$$

2. **Sequence Cleaning**:
     Parses each sequence, converting it to uppercase and stripping out whitespace, asterisks (`*`), numbers, periods, and any characters outside the standard IUPAC amino acid set {A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y}:
     $$s_{\text{clean}} = \{c \in s_{\text{raw}} \mid c \text{ is a standard amino acid}\}$$

3. **Length and Substring Filtering**:
     If filtering is enabled, a sequence is discarded if:
     $$\text{Length}(s_{\text{clean}}) < \text{MIN\_SEQ\_LENGTH} \quad \text{or} \quad \text{Length}(s_{\text{clean}}) > \text{MAX\_SEQ\_LENGTH}$$
     
     Or if the header contains the excluded substring:
     $$\text{Substring} \subseteq h_{\text{clean}}$$

4. **Serialization and Diagnostics**:
     Writes the sanitized sequences to the target FASTA file. It then analyzes the sequence length distribution and displays statistics (mean, median, standard deviation).

</details>

---

# 🧬 Generate Embeddings (`Generate_Embeddings.py`)

This script extracts sequence embeddings from pre-trained protein language models (like ESM-2, ESM-C, ProtBERT, or ProstT5). It maps residues to high-dimensional representation vectors and bundles them into HDF5 database files using compressed arrays to optimize speed and disk efficiency.

### 📥 Input

#### Sanitized FASTA File `INPUT_FASTA`
*   **Format**: Sanitized FASTA (`.fasta`).
*   **Created By**: `Sanitize_Sequences.py` (Sequence Sanitization utility).
*   **Description**: Output of the sequence sanitization process.

### ⚙️ Parameters

| Parameter | Description |
| :--- | :--- |
| Model Name **`MODEL_NAME`** | The protein language model architecture to use (e.g. `esmc_600m`, `esm2_t33_650m`, `esm2_t30_150m`, `protbert`, `prostt5`). |
| Saving Precision **`SAVING_MODE`** | The numeric precision format used to store vectors in HDF5 (`float16` or `float32`). `float16` is recommended to reduce disk space by 50% with negligible loss of accuracy. |

### 📤 Output

#### HDF5 Embedding Database
*   **Format**: HDF5 (`.h5`).
*   **Structure**:
    - `/{sanitized_header}`: Dataset of shape $L \times D$, where $L$ is sequence length and $D$ is model dimension.
    - `/headers`: Array of sequence headers.
    - `/seq_lens`: Array of sequence lengths.

<details markdown="1">
<summary><b>Algorithm Details</b></summary>

1. **Model Loader and Weight Cache**:
     Downloads and caches model weights from Hugging Face. Loads the transformer model and tokenizes the input.

2. **Hardware Target Selection**:
     Checks system capabilities and assigns tensor operations to the optimal accelerator:
     $$\text{Device} = \begin{cases} \text{cuda} & \text{if Nvidia GPU available} \\ \text{mps} & \text{if Apple Silicon available} \\ \text{cpu} & \text{otherwise} \end{cases}$$

3. **Residue Embedding Generation**:
     For each sanitized sequence $s$:
     - Tokenizes and formats the sequence with start/stop tokens:
       $$s_{\text{token}} = \langle\text{cls}\rangle \, s_1 \, s_2 \, \dots \, s_L \, \langle\text{eos}\rangle$$
     - Executes a forward pass without gradient calculations:
       $$H = \text{TransformerEncoder}(s_{\text{token}})$$
     - Extracts the final hidden states tensor $E \in \mathbb{R}^{(L+2) \times D}$ from the last hidden layer.
     - Slices off the start/stop boundary tokens, yielding the residue embedding matrix:
       $$E_{\text{residue}} = E_{1 \dots L} \in \mathbb{R}^{L \times D}$$

4. **HDF5 Database Compilation**:
     Saves each $E_{\text{residue}}$ dataset in the output HDF5 database using the sanitized header as the key. Slices values into the target precision (`float16` or `float32`). Writes a global `headers` index array and a `seq_lens` array to enable rapid database lookups.

</details>
