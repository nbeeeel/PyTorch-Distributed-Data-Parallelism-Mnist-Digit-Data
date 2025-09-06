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

# Set Matplotlib config directory to a writable location
export MPLCONFIGDIR=/tmp/matplotlib-$SLURM_JOBID

# Load CUDA/NCCL module (adjust based on your cluster)
# Example: module load cuda/11.8 nccl/2.12
# module load cuda/11.8  # Uncomment and adjust if needed

# Launch each rank
srun --mpi=none bash -c '
RANK=$SLURM_PROCID
WORLD_SIZE=$SLURM_NTASKS

# Set Ethernet interface for NCCL (These Interfaces are set for our own cluster)
case $RANK in
  0) export NCCL_SOCKET_IFNAME=ens1f1 ;;       # master-node
  1) export NCCL_SOCKET_IFNAME=ens1f1 ;;       # node01
  2) export NCCL_SOCKET_IFNAME=ens1f0np0 ;;    # node02
  3) export NCCL_SOCKET_IFNAME=ens1f1 ;;       # node03
  4) export NCCL_SOCKET_IFNAME=ens1f0 ;;       # node04
  5) export NCCL_SOCKET_IFNAME=ens1f0np0 ;;    # node05
esac

# Disable InfiniBand
export NCCL_IB_DISABLE=1

# Optional: set debug output for diagnostics
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1

# Export distributed environment variables
export RANK
export WORLD_SIZE
export MASTER_ADDR
export MASTER_PORT
export MPLCONFIGDIR

echo "RANK=$RANK on $(hostname) using interface $NCCL_SOCKET_IFNAME"

# Verify network interface
ip addr show $NCCL_SOCKET_IFNAME

# Verify connectivity to master
if [ $RANK -ne 0 ]; then
  ping -c 2 $MASTER_ADDR
  nc -zv $MASTER_ADDR $MASTER_PORT
fi

# Ensure port is open on master
if [ $RANK -eq 0 ]; then
  netstat -tuln | grep $MASTER_PORT
fi

python3 ./train.py
'
