🚀 Spatiotemporal GNN Performance Benchmark: Traffic Forecasting

💡 Project Overview

This repository hosts a reproducible benchmark comparing the performance and efficiency of two major Spatiotemporal Graph Neural Network (STGNN) paradigms for the Metr-LA Traffic Flow Forecasting task:

STGNN (TimeThenSpaceModel): A Decoupled architecture where temporal feature extraction precedes spatial aggregation.

DCRNN (Diffusion Convolutional Recurrent Neural Network): A Coupled architecture where time and space processing are interwoven at every step.

The project is built using the robust Torch Spatiotemporal (tsl) library, which integrates PyTorch, PyG, and PyTorch Lightning to streamline STGNN development and experimentation.

📊 Experimental Results and Performance Analysis

We trained and tested both models on the Metr-LA dataset, predicting traffic speed for 60 minutes (12 time steps) into the future.

Metric

STGNN (TimeThenSpace)

DCRNN (DCRNNModel)

Performance Advantage

Test MAE (Overall)

3.3683

3.3737

STGNN (Marginal)

Test MAPE (Overall)

0.0951

0.0939

DCRNN

Test MAE @ 60 mins (Long-Term)

4.1128

4.1013

DCRNN

Training Speed (it/s)

~70.96

~7.22

STGNN (Approx. 10x Faster)

🔍 Conclusion and Architectural Insights

Efficiency Winner: STGNN demonstrated a massive advantage in training speed. Its decoupled structure requires the costly Graph Neural Network (GNN) operation to be performed only once, significantly reducing computational overhead per training step, making it ideal for large-scale or iterative experiments.

Long-Term Dynamics: The coupled design of DCRNN allows its hidden state to continuously incorporate spatial information during temporal evolution. This dynamic, step-by-step spatial adjustment gives DCRNN a slight edge in the long-term 60-minute forecast accuracy.

🛠️ Environment Setup and Reproducibility Guide

Due to the sensitive nature of GNN libraries to PyTorch, PyG, and CUDA versions, it is crucial to set up the environment using the exact versions specified below to ensure project reproducibility.

1. System Prerequisites

Dependency

Version

Notes

Python

3.11.14

The precise Python version used for successful execution.

CUDA

12.1

Required for the locked PyTorch version (torch==2.5.1+cu121) and its compiled dependencies.

Hardware

NVIDIA GPU Recommended

Necessary for optimal training speed.

2. Create and Activate Virtual Environment

It is highly recommended to use a dedicated virtual environment (Conda or venv).

Using Conda (Recommended):

# 1. Create the environment with the specific Python version
conda create -n stgnn_env python=3.11.14

# 2. Activate the environment
conda activate stgnn_env


Using venv (If Conda is unavailable):

# 1. Create environment
python -m venv stgnn_env

# 2. Activate environment
# Windows:
.\stgnn_env\Scripts\activate
# Linux/macOS:
source stgnn_env/bin/activate


3. Install Dependencies

Use the provided requirements.txt file for bulk installation. This file locks all package versions, including the installation of the tsl library from a specific Git commit.

pip install -r requirements.txt


🚨 Crucial Compatibility Note:
If you encounter errors installing PyTorch/PyG dependencies (e.g., torch_scatter), ensure your system's CUDA setup is compatible with CUDA 12.1. You may need to consult the PyTorch documentation to adjust the requirements.txt versions if your system requires a different CUDA version.

4. Replicate the Experiment (Execution)

Launch Jupyter Notebook:

jupyter notebook


Run the Notebook:
Open the [TUTORIAL]_Spatiotemporal_Graph_Neural_Networks_with_tsl.ipynb file and execute all cells sequentially.

The Metr-LA dataset will be downloaded and processed automatically.

The models will train, save the best checkpoint, and run the final comparative evaluation.
