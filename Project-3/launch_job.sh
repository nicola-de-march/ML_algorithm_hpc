#!/bin/bash -l
#SBATCH -N 2 # Number of nodes
#SBATCH --ntasks-per-node=1 # Number of tasks per nod
#SBATCH -c 64
#SBATCH --time=0-02:00:00
#SBATCH -p batch
#SBATCH -J cnn_training
#SBATCH --output=cnn_training.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nicola.demarch.001@student.uni.lu

conda activate Project-3
mpiexec -n 50 python cnn_mpi.py
