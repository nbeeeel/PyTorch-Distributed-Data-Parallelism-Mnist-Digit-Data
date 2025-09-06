# 🚀 Distributed MNIST Training with TensorFlow & PyTorch (MultiWorkerMirroredStrategy + SLURM)

This repository demonstrates **distributed deep learning** on the **MNIST digit classification task** using:

- **TensorFlow’s `MultiWorkerMirroredStrategy`** (for MNIST dataset handling + demonstration)
- **PyTorch Distributed Data Parallel (DDP)** backend concepts (integration-ready)
- **NCCL backend for GPU communication**
- **SLURM job scheduler** for multi-node orchestration

The project is modularized for clarity, extensibility, and easy adaptation to other datasets or models.

---

## 📂 Repository Structure

```
.
├── config/
│   └── env_setup.py       # Environment + TF_CONFIG setup
├── data/
│   └── dataset.py         # MNIST dataset loader (download + preprocess)
├── models/
│   └── simple_cnn.py      # Simple CNN model for digit classification
├── training/
│   ├── strategy.py        # Distributed strategy initialization
│   └── trainer.py         # Training loop (with gradient updates)
├── train.py               # Entry point (orchestrates everything)
├── run_slurm.sh           # SLURM job script for multi-node execution
└── README.md              # Project documentation
```

---

## ✨ Features

- ✅ Multi-node **data parallel training**
- ✅ **NCCL communication backend** for GPU communication
- ✅ Modularized codebase (datasets/models/training loop)
- ✅ SLURM job script for running across a **GPU cluster**
- ✅ Barrier sync + retry logic for robustness
- ✅ Uses **MNIST dataset** (auto-downloaded)

---

## ⚙️ Installation

Clone this repository:

```bash
git clone https://github.com/nbeeeel/PyTorch-Distributed-Data-Parallelism-Mnist-Digit-Data.git
cd PyTorch-Distributed-Data-Parallelism-Mnist-Digit-Data
pip install -r requirements.txt
```

### Requirements

- **Python 3.8+**
- **TensorFlow 2.11+**
- **PyTorch 1.12+**
- **CUDA 11.8 (or matching cluster version)**
- **NCCL 2.12+**

---

## 📊 Model Architecture

The **SimpleCNN** defined in `models/simple_cnn.py`:

- Flatten input (28×28×1)
- Dense layer (128 units, ReLU)
- Dense layer (10 units, logits for classification)

---

## 🧩 How It Works

### 🔹 Step 1. Environment Setup
Each worker configures:
- `TF_CONFIG` with cluster topology
- NCCL communication interface
- GRPC keepalive options

### 🔹 Step 2. Strategy Initialization
- Uses **`MultiWorkerMirroredStrategy`** with **NCCL backend**
- Retries initialization on failure (up to 3 times)

### 🔹 Step 3. Dataset Loading
- Downloads and preprocesses **MNIST**
- Normalizes to `[0,1]`
- Builds shuffled + batched `tf.data.Dataset`

### 🔹 Step 4. Training Loop
- Each replica runs `train_step` with gradient tape
- Loss is reduced across replicas
- Leader rank (`rank=0`) logs epoch loss

---

## 🖥️ Running with SLURM

A sample SLURM job script (`run_slurm.sh`) is included:

```bash
#!/bin/bash
#SBATCH --job-name=nccl_dist
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --nodelist=node01,node02,node03,node04,node05
#SBATCH --time=01:00:00
#SBATCH --output=nccl_%j.out
#SBATCH --error=nccl_%j.err

# Master node address and port
export MASTER_ADDR=192.168.20.15
export MASTER_PORT=29500

export MPLCONFIGDIR=/tmp/matplotlib-$SLURM_JOBID

# Example: module load cuda/11.8 nccl/2.12
# module load cuda/11.8

srun --mpi=none bash -c '
RANK=$SLURM_PROCID
WORLD_SIZE=$SLURM_NTASKS

case $RANK in
  0) export NCCL_SOCKET_IFNAME=ens1f1 ;;
  1) export NCCL_SOCKET_IFNAME=ens1f1 ;;
  2) export NCCL_SOCKET_IFNAME=ens1f0np0 ;;
  3) export NCCL_SOCKET_IFNAME=ens1f1 ;;
  4) export NCCL_SOCKET_IFNAME=ens1f0 ;;
  5) export NCCL_SOCKET_IFNAME=ens1f0np0 ;;
esac

export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1

export RANK
export WORLD_SIZE
export MASTER_ADDR
export MASTER_PORT
export MPLCONFIGDIR

python3 ./train.py
'
```

### ✅ Submit job:

```bash
sbatch run_slurm.sh
```

Logs are saved as `nccl_<JOBID>.out` and `nccl_<JOBID>.err`.

---

## 📈 Example Output

```text
RANK 0 on node01: Setting TF_CONFIG: {...}
RANK 1 on node02: Connected to 192.168.20.15:29500
[Epoch 0] Loss: 0.2751
[Epoch 1] Loss: 0.1927
...
```

---

## 📌 Notes & Tips

- Ensure **all nodes have identical environments** (same TF/PyTorch/CUDA/NCCL versions).
- Adjust **`NCCL_SOCKET_IFNAME`** to match your cluster’s network interfaces.
- Use `NCCL_DEBUG=INFO` for diagnostics.
- Modify **`train.py`** to plug in new datasets or models.

---

## 🔮 Future Extensions

- Add checkpoint saving and resume support
- Extend to CIFAR-10 / ImageNet
- Replace SimpleCNN with deeper models (ResNet, EfficientNet)
- Add evaluation pipeline

---

## 📜 License

This project is released under the **MIT License**.

---

## 🤝 Contributing

Contributions are welcome!  
Open an **issue** or **pull request** with improvements or bug fixes.

---
