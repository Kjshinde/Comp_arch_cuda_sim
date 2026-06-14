# Computer Architecture & Simulation Portfolio

Welcome to my repository for Computer Architecture and Simulation projects, developed as part of **ASU EEE 520 (Computer Architecture)**. This repository showcases my hands-on experience with architectural simulation using **gem5** and high-performance parallel computing using **CUDA**.

## 📌 Repository Structure & Projects

### 1. `gem5-LRU-IPV-Cache/` (gem5 Cache Replacement Policy)
- **Description:** Implemented the **LRU Insertion Promotion Vector (IPV)** cache replacement policy within the gem5 simulator architecture.
- **Key Features:**
  - Modified the gem5 caching mechanism by adding custom `LRUIPVRP` classes (`lru_ipv.cc`, `lru_ipv.hh`).
  - Implemented shared recency states and per-block replacement data to efficiently track block usage.
  - Integrated with gem5's Python configuration scripts (`CacheConfig.py`, `Options.py`, `ReplacementPolicies.py`).
- **Technologies:** C++, Python, gem5

### 2. `CUDA-Deep-Learning/` (Custom CUDA Neural Network Kernels)
- **Description:** Built a custom deep learning framework from scratch by writing custom CUDA C++ kernels for forward and backward passes.
- **Sub-Projects:**
  - `Basic-Kernels/`: Initial implementation of forward and backward passes for Linear Layers and ReLU Activations directly in CUDA (`linear_forward_kernel.cu`, `relu_backward_kernel.cu`).
  - `Python-Wrapped-Kernels/`: A more advanced framework bridging Python and the compiled CUDA C++ code using `ctypes`. This allows efficiently managing memory allocation and data transfers between the CPU and GPU while training the model via Python scripts (`linear_wrapper.cu`, `relu_wrapper.cu`, `training.py`).
- **Technologies:** CUDA C++, Python, GPU Memory Management

### 3. `Architecture-Reports/` (Research and Design)
- **Description:** Contains detailed design documents and reports (`G19-Project1-Submission.pdf`, `G19-Project3-Submission.pdf`) outlining architectural evaluations and simulation results for various CPU and pipeline architectures.

### 4. `Archive.zip`
- Contains the original submission zip files for reference (password protected).

## 🛠️ Technical Skills Demonstrated
* **System Simulation:** gem5
* **High-Performance Computing:** CUDA, Parallel Processing, Grid/Block Thread Optimization
* **Languages:** C++, Python, C
* **Concepts:** Cache Replacement Policies (LRU IPV), Memory Hierarchy, Backpropagation, GPU Memory Management (Host-to-Device/Device-to-Host transfers).

## 🚀 Getting Started

### Prerequisites
- **gem5 Simulator:** Required to build and run the LRU-IPV cache policy.
- **CUDA Toolkit:** Required to compile the `.cu` files using `nvcc`.
- **Python 3.x:** Required to run the training scripts and gem5 configurations.

### Running the CUDA Neural Network (`Python-Wrapped-Kernels/`)
1. Compile the CUDA code into a shared library:
   ```bash
   cd CUDA-Deep-Learning/Python-Wrapped-Kernels
   nvcc -Xcompiler -fPIC -shared -o libcustom_kernels.so linear_wrapper.cu relu_wrapper.cu
   ```
2. Execute the training script:
   ```bash
   python training.py
   ```
*(Note: Ensure the shared object name in the Python file matches your compiled output).*

---
*Created by Krutyanjay Shinde for ASU EEE 520.*

**Password for the zip: Name of the house you grew up in (ALLCAPS)**
