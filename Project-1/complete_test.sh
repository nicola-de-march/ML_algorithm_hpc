#!/bin/bash -l
#SBATCH -N 2 # Number of nodes
#SBATCH --ntasks-per-node=1 # Number of tasks per node
#SBATCH -c 100
#SBATCH --time=0-04:00:00
#SBATCH -p batch
#SBATCH -J knn_training
#SBATCH --output=knn_training.out
#SBATCH --mail-user nicola.demarch.001@student.uni.lu # Mail to send notifications
#SBATCH --mail-type BEGIN,END,FAIL # Type of notifications to send

module load lang/SciPy-bundle
unset I_MPI_PMI_LIBRARY

process_counts=(20 40 50 100 200)

for count in "${process_counts[@]}"; do
    for i in {1..30}; do  
        echo "Running with $count processes (Iteration $i)"
        mpiexec -n $count python test_knn.py
    done
done
