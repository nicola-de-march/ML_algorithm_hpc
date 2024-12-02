#!/bin/bash -l
#SBATCH -N 2 # Number of nodes
#SBATCH --ntasks-per-node=1 # Number of tasks per nod
#SBATCH -c 64
#SBATCH --time=0-04:00:00
#SBATCH -p batch
#SBATCH -J cnn_training_jax
#SBATCH --output=cnn_training_jax.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nicola.demarch.001@student.uni.lu

conda activate Project-3

for input in 5 10 15 20 25
do
  for i in {1..5}
  do
    echo "Running serial version with input $input"
    mpiexec -n $input python cnn_mpi_jax.py $input
  done
done