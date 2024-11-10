# Accelartion of KNN using MPI
**Author:** Nicola De March

This document explains how the work was conducted.

In the directory you can find the following files:
- `KNNClassifier.py`- This file contains the class that implements the KNN algorithm, accelerated with MPI.
- `test_knn.py` - This file initializes the computation.
- `complete_test.sh` – This script launches the computation using different numbers of processors. Specifically, it runs the computation 30 times for each processor count (20, 40, 50, 100, and 200 cores).
  > **Note**: Running this script takes a significant amount of time.
- `mpi_knn_results.csv` - This file stores all execution times along with other parameters, such as dataset size and test set size.
- `time_analysis.ipynb` This notebook loads the CSV file into a pandas DataFrame and computes statistics like **average** time and **standard deviation**. `time_analysis.html` is the pre-executed notebook, displaying plots and tables.
- `launch_job.sh` launchs a single job with 200 cores.


