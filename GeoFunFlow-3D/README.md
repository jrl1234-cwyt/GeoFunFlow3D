# GeoFunFlow-3D

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

This repository contains the official PyTorch implementation for the paper: **GeoFunFlow-3D: A Physics-Guided Generative Flow Matching Framework for High-Fidelity 3D Aerodynamic Inference over Complex Geometries** (Submitted to the *Journal of Computational Physics*, JCP).

This codebase provides the core network architectures, training scripts, and JCP-style visualization tools required to reproduce the generation of 3D aerodynamic flow fields (e.g., NASA Rotor37 and BlendedNet) under extremely sparse sampling conditions.

## 🚀 Overview
GeoFunFlow-3D addresses the severe VRAM bottlenecks and high-frequency spectral bias commonly encountered in 3D physics-informed deep learning. By deeply integrating a No-AD (Automatic Differentiation-free) discrete differential engine, a SATO (Topology-Aware Super-Resolution) multi-scale module, and a Variational Homotopy Scheduling strategy, this framework enables highly robust **Generative Flow Matching** on unstructured 3D point clouds.

### ✨ Key Highlights
* **Spectral Stability:** A No-AD discrete difference engine that mitigates gradient stiffness and solves high-frequency spectral bias without exploding VRAM.
* **Spatial Consistency:** The SATO module enforces local physical consistency in critical micro-regions (like shock waves and boundary layers) using deterministic phase-field hard masking.
* **Temporal Convergence:** Variational Homotopy Scheduling rebuilds the generation path via optimal transport theory, ensuring global nonlinear optimization stability.

## 📁 Repository Structure
- `models/`: Core network architectures (`dit_model_3d.py`, `fno_modules_unified.py`, `gino_encoder_3d.py`, `hybrid_decoder_unified.py`).
- `utils/`: Utility libraries, including physical hard constraints, thermodynamic loss computation (`physics_unified.py`), and weight scheduling (`loss_schedulers_3d.py`).
- `dataset_unified.py`: Dataloader designed to handle `.vtp` and `.npz` format point cloud datasets.
- `preprocess_data.py`: Data preprocessing script for computing the exact Signed Distance Field (SDF).
- `train_fae.py` & `train_flow.py`: Two-stage optimization training scripts (Feature Auto-Encoder warm-up & Flow Matching evolution).

## ⚙️ Installation

Clone this repository and install the required dependencies:

```bash
git clone [https://github.com/jrl1234-cwyt/GeoFunFlow3D.git](https://github.com/jrl1234-cwyt/GeoFunFlow3D.git)
cd GeoFunFlow3D
pip install -r requirements.txt
