#!/bin/bash -l
#SBATCH -N 2 # Number of nodes
#SBATCH --ntasks-per-node=1 # Number of tasks per nod
#SBATCH -c 64
#SBATCH --time=0-03:00:00
#SBATCH -p batch
#SBATCH -J cnn_training_serial
#SBATCH --output=cnn_training_serial.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nicola.demarch.001@student.uni.lu

conda activate Project-3

for input in 5 10 15 20 25
do
  echo "Running serial version with input $input"
  python P3.py $input
done

