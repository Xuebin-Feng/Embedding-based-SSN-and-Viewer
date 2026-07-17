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

---

# ✂️ Embedding Cropping (`Embedding_Cropping.py`)

This script produces embeddings for cropped/partial sequences by slicing them directly out of an existing full-sequence embedding database, instead of embedding the cropped fragment in isolation. Protein language models compute every residue's representation using full self-attention context, so directly embedding a short fragment yields a different (context-impoverished) vector than the same residues would get inside their native full-length sequence. This script never re-runs the language model — it only reads a full-sequence HDF5 database (produced by `Generate_Embeddings.py`) and slices out the requested residue range for each cropped sequence, preserving the full-context representation.

### 📥 Input

#### Full Embedding Database `INPUT_EMBED`
*   **Format**: HDF5 embedding database (`.h5`).
*   **Created By**: `Generate_Embeddings.py` (Embedding Generation utility), run on `FULL_FASTA`.

#### Full Sequence Set `FULL_FASTA`
*   **Format**: Standard FASTA (`.fasta`).
*   **Description**: The full-length sequences that `INPUT_EMBED` was generated from. May contain more sequences than `CROPPED_FASTA` needs — extras are ignored.

#### Cropped Sequence Set `CROPPED_FASTA`
*   **Format**: Standard FASTA (`.fasta`).
*   **Description**: Partial sequences to produce contextual embeddings for. Each header must also appear in `FULL_FASTA` and `INPUT_EMBED`, and each sequence must be an exact contiguous substring of its full-length counterpart.

### ⚙️ Parameters

This script does not require additional configuration parameters — behavior is fully determined by the three input files above.

### 📤 Output

#### HDF5 Embedding Database
*   **Format**: HDF5 (`.h5`), named `{CROPPED_FASTA}_[{model_name}]_embeddings.h5` — identical in structure to what `Generate_Embeddings.py` would produce if run directly on `CROPPED_FASTA`, so it is a drop-in input for downstream tools (`Embedding_PWA.py`, `Embedding_SSEARCH.py`, `Embedding_MSA.py`, etc.).
*   **Structure**:
    - `/embeddings/{sanitized_header}`: Dataset of shape $L_{\text{crop}} \times D$, sliced from the full-length embedding.
    - `/headers`: Array of resolved cropped-sequence headers.
    - `attrs["model_name"]`, `attrs["num_sequences"]`.

<details markdown="1">
<summary><b>Algorithm Details</b></summary>

1. **Header Correspondence**:
     For each cropped sequence $s_{\text{crop}}$ with header $h$, locates the full sequence $s_{\text{full}}$ and full embedding matrix $E_{\text{full}} \in \mathbb{R}^{L_{\text{full}} \times D}$ sharing the same header $h$ in `FULL_FASTA` / `INPUT_EMBED`. Headers absent from either are reported and skipped.

2. **Consistency Check**:
     Verifies that the full embedding's row count matches the full sequence's length:
     $$L_{\text{full}} \overset{?}{=} \text{Length}(s_{\text{full}})$$
     A mismatch indicates `INPUT_EMBED` does not actually correspond to `FULL_FASTA` for that header, and the entry is skipped rather than sliced incorrectly.

3. **Offset Resolution**:
     Finds the position of the cropped sequence within its full parent via exact substring search:
     $$\text{offset} = \arg\min \{ i : s_{\text{full}}[i : i+L_{\text{crop}}] = s_{\text{crop}} \}$$
     If the crop occurs more than once, the first occurrence is used and a warning is logged. If it is not found at all, the header is skipped and reported.

4. **Context-Preserving Slice**:
     Because the residue embeddings in $E_{\text{full}}$ already account for full sequence context, slicing is a simple index range with no recomputation:
     $$E_{\text{crop}} = E_{\text{full}}[\text{offset} : \text{offset} + L_{\text{crop}}]$$

5. **HDF5 Database Compilation**:
     Writes each $E_{\text{crop}}$ into the output database keyed by the sanitized cropped header, alongside a `headers` index array and `model_name`/`num_sequences` metadata — matching the standard embedding database structure.

</details>
