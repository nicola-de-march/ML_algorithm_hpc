#!/bin/bash -l
#SBATCH -N 2 # Number of nodes
#SBATCH --ntasks-per-node=1 # Number of tasks per nod
#SBATCH -c 64
#SBATCH --time=0-01:00:00
#SBATCH -p batch
#SBATCH -J knn_training
#SBATCH --output=knn_training.out

module load lang/SciPy-bundle
unset I_MPI_PMI_LIBRARY
mpiexec -n 200 python test_knn.py
