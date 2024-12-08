# 3 Project: image classification 
For the third project of the course, we dealt with the training of a model for image classification. The used dataset is the Minst dataset.

Starting from the serial implementation of the code contained in [P3.py](./P3.py), we slightly modified it and implemented an MPI version [cnn_mpi.py](./cnn_mpi.py) and MPI/Jax version [cnn_mpi_jax](./cnn_mpi_jax.py).

In the versions with MPI, each process manages more than one image and implement the stochastic gradient method to update the kernel.

To compare the performance of the three approaches, we executed the implementations for 5, 10, 15, and 20 images and collected the time statistics in three CSV files: ([time_analysis_serial.csv](./time_analysis_serial.csv)), ([time_analysis_mpi.csv](./time_analysis_mpi.cs)) and ([time_analysis_jax.csv](./time_analysis_jax.cs)).

The notebook ([time_analysis.ipynb](./time_analysis.ipynb)) contains the graph with the performance comparison. 


## How to execute on AION-cluster
1) Allocate a CPU node
```bash
si -t 30 --nodes 1 --ntasks-per-node 20 --cpus-per-task 1
```

2) Create and activate the environment (If already existing, otherwise create it)
```bash
conda env -n environment.yml
conda activate
```

3) Load the necessary tools
```bash
module load mpi/OpenMPI
```

4) Execute the code

Serial version
```bash
python P3.py
```
MPI version
```bash
mpiexec -n 20 python cnn_mpi.py
```
MPI version with Jax
```bash
mpiexec -n 20 python cnn_mpi_jax.py
```


## Contributors
- Nicola De March
- Giorgio Bettonte
