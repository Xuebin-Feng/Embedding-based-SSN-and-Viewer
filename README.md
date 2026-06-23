# Embedding-based Sequence Similarity Network (SSN) Viewer

[![Python Version](https://img.shields.io/badge/python-%3E3.10-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-windows%20%7C%20linux%20%7C%20macOS-lightgrey.svg)](https://github.com/)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)
[![Framework PyQt6](https://img.shields.io/badge/UI-PyQt6-orange.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![Render VisPy](https://img.shields.io/badge/Render-VisPy-red.svg)](https://vispy.org/)

The **Embedding-based SSN Viewer** is an interactive, high-performance graphical application designed to streamline the generation, visualization, and analysis of both traditional and embedding-based Sequence Similarity Networks (SSNs). By integrating **Multiple Sequence Alignments (MSAs)** directly into network exploration, the viewer bridges macroscopic sequence relationships with microscopic residue-level conservation, providing a comprehensive, multi-scale view of the protein sequence space.

---

## ⚠️ Important Note

1. **Cross-Platform Support**: Linux & macOS support is currently under active development and in progress.
2. **Work in Progress**: This documentation and project structure are still updating.
3. **Recommended Hardware**: An **NVIDIA GPU** is highly recommended for CUDA acceleration of embeddings and layout solvers.

---

## 📸 Preview

*Insert an animated GIF or video here showing the interactive PyQt6 viewport in action, displaying node selections, zooming, and color mappings.*
```markdown
![SSN Viewer UI Demonstration](docs/assets/ssn_viewer_demo.gif)
```

---

## 🚀 Key Features

*Work in Progress*

---

## 🧬 System Workflow

*Work in Progress*

---

## ⚙️ Installation Steps



1. **Clone the repository:**
   Download or clone the repository to your computer.

2. **Set up the environment:**

   **Windows**:
   Double-click `SSN_Tools.bat` or `SSN_Viewer.bat`. A self-contained virtual Python environment will be created at the project root automatically.

   **Linux/macOS (In Progress)**:
   Open your terminal and run the following:
     
   1. Navigate to the project directory:
   ```bash
   cd Sequence_Similarity_Network_Viewer
   ```
     
   2. Grant execution permissions to the `.sh` script files:
   ```bash
   chmod +x *.sh
   ```
     
   3. Execute the tools script to initialize the virtual environment and install dependencies:
   ```bash
   ./SSN_Tools.sh
   ```

---

## 💻 Usage

*Work in Progress*

---

## 📂 Repository Structure

```directory
Sequence_Similarity_Network_Viewer/
│
├── SSN_Viewer.py            # Main PyQt6 / VisPy desktop visualization application
├── SSN_Tools.py             # CLI utility for generating network data & computing layouts
├── SSN_Config.py            # Central configuration file for thresholds, colors, and models
├── SSN_Utils.py             # Shared utility functions (IO, math helper, parsing)
│
├── Alignment_Manager.py     # Pairwise and multiple sequence alignment runner
├── Command_Engine.py        # Pipeline execution coordinator
├── Detect_GPU.py            # GPU hardware check script
│
├── Layout_Engine_UMAP.py             # Coordinates projection using UMAP on ESM embeddings
├── Layout_Engine_SSN_MonteCarlo.py   # Energy minimization layout solver
├── Layout_Engine_SSN_MolecularDynamics.py # Force-directed spring-electrical layout solver
│
├── VR_Viewer (work in progress)/     # VR-specific configuration, scripts, and Unity Application
│   ├── SSN_VR_Viewer.py     # Python runner coordinating network streams to VR
│   └── VR_App/              # Precompiled Unity executable for VR headsets
│
├── Input_Files/             # Put your raw input sequence FASTA files here
├── Embeddings/              # Directory where ESM protein embeddings are cached
└── Results/                 # Visual outputs, exported graphs, and layouts
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to open Issues or submit Pull Requests to enhance layout performance, UI responsiveness, or VR interactions.

## 📄 License

This project is licensed under the GNU GPL v3 License - see the [LICENSE](LICENSE) file for details.
