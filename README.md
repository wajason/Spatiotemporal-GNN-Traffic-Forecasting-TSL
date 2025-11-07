# 🚦 Spatiotemporal GNNs for Traffic Forecasting — Reproducible Benchmark & Deep Dive
[![GitHub Stars](https://img.shields.io/github/stars/wajason/Spatiotemporal-GNN-Traffic-Forecasting-TSL?style=for-the-badge&logo=github&color=4C8EDA)](https://github.com/wajason/Spatiotemporal-GNN-Traffic-Forecasting-TSL/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/wajason/Spatiotemporal-GNN-Traffic-Forecasting-TSL?style=for-the-badge&logo=github&color=4C8EDA)](https://github.com/wajason/Spatiotemporal-GNN-Traffic-Forecasting-TSL/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/wajason/Spatiotemporal-GNN-Traffic-Forecasting-TSL?style=for-the-badge&color=4C8EDA)](https://github.com/wajason/Spatiotemporal-GNN-Traffic-Forecasting-TSL/issues)
[![GitHub License](https://img.shields.io/github/license/wajason/Spatiotemporal-GNN-Traffic-Forecasting-TSL?style=for-the-badge&color=4C8EDA)](./LICENSE)

[![Python](https://img.shields.io/badge/Python-3.11.14+-4C8EDA?style=for-the-badge&logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)]()
[![PyG](https://img.shields.io/badge/PyG-2.7.0-4C8EDA?style=for-the-badge&logo=pyg&logoColor=white)]()
[![TSL Framework](https://img.shields.io/badge/TSL-Library-4C8EDA?style=for-the-badge)](https://torch-spatiotemporal.readthedocs.io/)

[![Open in GitHub Codespaces](https://img.shields.io/badge/Run%20in-Codespaces-4C8EDA?style=for-the-badge&logo=github)](https://codespaces.new/wajason/Spatiotemporal-GNN-Traffic-Forecasting-TSL)


This repository provides a clean and fully reproducible benchmark comparing two influential classes of **Spatiotemporal Graph Neural Networks** for traffic forecasting:

- **Time-then-Space STGNN** (Temporal compression followed by graph-based spatial aggregation)
- **DCRNN** (Diffusion Convolutional Recurrent Neural Network; joint spatiotemporal processing)

The project highlights how **model design philosophy** impacts forecasting behavior, computational efficiency, and generalization across prediction horizons — especially in real-world traffic systems such as **Metr-LA**.

This is not just a "run-the-code" repo — it is a **concept-to-result walkthrough** designed for:
- GNN / Spatiotemporal learning practitioners
- Applied ML researchers
- Students learning STGNN model design
- Engineers preparing for real deployment on limited compute

If this project helps you — ⭐ **Please consider starring the repo!**

---

## 🔍 Key Contributions

| Contribution | Description |
|-------------|-------------|
| **Clear architectural comparison** | Directly contrasts *decoupled* (STGNN) vs. *coupled* (DCRNN) spatiotemporal learning strategies |
| **Parameter efficiency study** | Shows how smaller STGNNs can achieve comparable accuracy with ~50% fewer parameters |
| **Fully reproducible pipeline** | Environment, dataset, training, logging, and evaluation all included |
| **Interpretable performance insights** | Short-horizon vs Long-horizon forecast differences are discussed instead of just reported |

---

## 📊 Results Summary (Metr-LA, 60-min Forecasting)

| Model | Params | Train Speed | MAE (Avg) | MAE @ 60min | Notes |
|------|--------|-------------|-----------|-------------|-------|
| **STGNN (TimeThenSpace)** | 19.3K | **~10× faster** | **3.368** | 4.113 | Strong short-term temporal representation |
| **DCRNN** | 42.4K | Slower | 3.374 | **4.101** | Strong long-horizon stability via recurrent diffusion |

**Interpretation**  
- STGNN excels in **temporal abstraction efficiency**, making it ideal for real-time or resource-constrained deployments.  
- DCRNN retains **slightly better long-term temporal alignment** due to recurrent spatial-state integration.

---

## 🧱 Architecture Insight (Why This Matters)
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/01a99494-2eef-4eef-ba5d-1bd5bd860805" />

**STGNN ("Time → Space")**  
Compresses temporal dynamics first → applies graph convolution once → *computationally lean*.  
Best when **time patterns dominate** signal structure.

**DCRNN (Time ↔ Space Coupled)**  
Applies graph diffusion inside every recurrent step → *cost heavier but spatially adaptive*.  
Best when **network topology strongly influences future states**.

---

## 🧪 Training & Reproducibility

This project uses **Torch Spatiotemporal (tsl)** with:
- PyTorch Lightning for training loops
- PyTorch Geometric for GNN kernels
- Automatic data downloading & preprocessing

**Environment is fully version-locked** in `requirements.txt`  
→ This ensures repeatable runs — no dependency roulette.

---

## 🛠️ Environment Setup and Reproducibility Guide

Due to the sensitive nature of GNN libraries to PyTorch, PyG, and CUDA versions, it is crucial to set up the environment using the **exact versions** specified below to ensure project reproducibility.

### 1. System Prerequisites

| **Dependency** | **Version** | **Notes** |
| :--- | :--- | :--- |
| **Python** | `3.11.14` | The precise Python version used for successful execution. |
| **CUDA** | `12.1` | Required for the locked PyTorch version (`torch==2.5.1+cu121`) and its compiled dependencies. |
| **Hardware** | NVIDIA GPU Recommended | Necessary for optimal training speed. |

### 2. Create and Activate Virtual Environment

It is highly recommended to use a dedicated virtual environment (Conda or venv).

#### **Using Conda (Recommended):**

```bash
# 1. Create the environment with the specific Python version
conda create -n stgnn_env python=3.11.14

# 2. Activate the environment
conda activate stgnn_env
```

#### **Using venv (If Conda is unavailable):**

```bash
# 1. Create environment
python -m venv stgnn_env

# 2. Activate environment
# Windows:
.\stgnn_env\Scripts\activate
# Linux/macOS:
source stgnn_env/bin/activate
```
### 3. Install Dependencies

Use the provided requirements.txt file for bulk installation. This file locks all package versions, including the installation of the tsl library from a specific Git commit.

```bash
pip install -r requirements.txt
```
🚨 Crucial Compatibility Note: If you encounter errors installing PyTorch/PyG dependencies (e.g., torch_scatter), ensure your system's CUDA setup is compatible with CUDA 12.1. You may need to consult the PyTorch documentation to adjust the requirements.txt versions if your system requires a different CUDA version.

### 4. Replicate the Experiment (Execution)
```bash
jupyter notebook
```

Run the Notebook: Open the [TUTORIAL]_Spatiotemporal_Graph_Neural_Networks_with_tsl.ipynb file and execute all cells sequentially.  
The Metr-LA dataset will be downloaded and processed automatically.  
The models will train, save the best checkpoint, and run the final comparative evaluation.


## ⭐ If You Find This Useful

This project took effort to organize & document clearly.  
If it helps you learn, research, or build your own system:

→ **Please give the repository a Star**  
It really helps visibility and motivates future updates ❤️
