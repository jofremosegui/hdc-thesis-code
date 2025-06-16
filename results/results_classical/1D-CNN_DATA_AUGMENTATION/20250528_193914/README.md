# Brain-Inspired Hyperdimensional Computing for Wearable Biosignal Classification

This repository contains the full codebase and experimentation pipeline for the Bachelor's Thesis:  
**"Exploring Brain-Inspired Hyperdimensional Computing for Resource-Efficient Wearable Biosignal Classification"**  
carried out at the University of California, Irvine and the Universitat Politècnica de Catalunya.

The project benchmarks classical, symbolic (HDC), and hybrid models for time series classification using physiological signals from wearable sensors. The focus is on interpretability, computational efficiency, and real-time deployability on edge devices.

---

## 📁 Repository Structure

```
benchmarking/                     # Inference time, memory, and runtime benchmarks
    runtime_memory_benchmarking/  # Deployment RAM profiling scripts per model
    BENCHMARKS_*.ipynb            # Notebooks summarizing deployability per model

eda_and_plotting/                 # EDA + final plots for figures (radar, Sankey, etc.)

experimentation/                 # Exploration and experiments

models/                          # Notebooks with model architecture and training logic
    classical_models/            # CNN, GRU, LSTM, Bi-LSTM
    hdc_models/                  # BSC, HRR, FHRR, MAP (standalone HDC)
    hybrid_models/               # MAP encoder + classical classifier (SVM, MLP, etc.)

results/                         # Final metrics, configs, and logs for each model group
    results_classical/
    results_hdc/
    results_hybrid/

torchhd/                         # Custom TorchHD fork used for symbolic modeling
    HDCArena/                    # Extended symbolic methods and evaluation tools
    datasets/, tensors/, tests/  # Internal modules and dataset wrappers
```

---

## 🚀 Project Summary

The thesis addresses the problem of intoxication detection using motion and heart-rate signals from wearable sensors. It benchmarks multiple model families under a common evaluation pipeline:

- ✅ **Classical Deep Models:** CNN, LSTM, GRU, Bi-LSTM
- ✅ **Symbolic Models (HDC):** BSC, HRR, FHRR, MAP
- ✅ **Hybrid Architectures:** MAP encoding + classical classifiers (SVM, MLP, XGBoost, LightGBM)

Each model is evaluated not only on classification metrics (accuracy, F1, drunk accuracy) but also **deployability metrics** like inference time, RAM usage, and energy consumption.

---

## 📦 Environment Setup

Ensure Python 3.11+ is installed. Use the following to create a clean environment:

```bash
git clone https://github.com/yourname/hdc-wearable-classification.git
cd hdc-wearable-classification
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt  
```

---

## 📊 How to Run Experiments

1. **Model Training & Evaluation:**
   - Run any notebook in `models/` or `experimentation/` to train and test a specific architecture.
   - For example:
     ```
     cd models/hdc_models/
     jupyter notebook HDC_MAP.ipynb
     ```

2. **Benchmarking:**
   - Deployment metrics are generated via:
     ```
     cd benchmarking/runtime_memory_benchmarking
     python memory_benchmark_HDC_MAP.py
     ```

3. **Figures and Analysis:**
   - Final plots are in `eda_and_plotting/PLOTS_AND_THINGS.ipynb`

---

## 📈 Results Overview

- **Best Overall:** MAP + LightGBM
- **Fastest Inference:** CNN
- **Most Lightweight and Accurate:** MAP (standalone HDC)
- **Most Accurate:** MAP + XGBoost
- See all metrics in `results/` and final comparison tables in the thesis appendix.

---

## 🧠 TorchHD Integration

This repository uses a **custom fork of TorchHD** (see `/torchhd`) to support advanced HDC operations and hybrid pipelines.


---

## 📝 Citation

If you find this repository useful, please cite the corresponding thesis or get in touch via email.

---

**Author:**  
Jofre Moseguí Monterde  
Balsells Fellow @ UC Irvine | UPC FIB
