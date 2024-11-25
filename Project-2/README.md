# Ensemble of Forecasters
For the second project of the course, we dealt with an ensemble of forecasters.

Starting from the serial implementation of the code contained in ensamble_of_forecasting.py, we slightly modified it and implemented an MPI version: mpi_ensamble_of_forecasting.py.

In the MPI version, each process manages a portion of forecasters fed with random parameters generated according to its rank.

At the end, RANK 0 performs a GatherV operation to collect the results of each process.

To compare the performance of the two approaches, we executed the implementations for 128, 256, 512, and 1024 forecasters and collected the time statistics in two CSV files: (results_serial_Forecaster.csv), (results_128MPI_Forecaster.csv). For the MPI version, we used 128 processes.

The notebook (time_stats.ipynb) contains the graph with the performance comparison. We can see that while the serial version grows linearly with the number of forecasters, the execution time for the MPI version using 128 processes remains under control.


## How to execute on AION-cluster
1) Allocate a CPU node
```bash
si -t 30 --nodes 1 --ntasks-per-node 128 --cpus-per-task 1
```

2) Activate the environment (If already existing, otherwise create it)
```bash
micromamba activate ds
```

3) Install and load the necessary tools
```bash
pip install jax
module load mpi/OpenMPI
```

4) Execute the code

Serial version
```bash
python ensamble_of_forecasting.py
```
MPI version
```bash
mpiexec -n 128 python mpi_ensamble_of_forecasting.py
```


## Contributors
- Nicola De March
- Giorgio Bettonte
