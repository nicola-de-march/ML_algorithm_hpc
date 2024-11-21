from KNNClassifier import KNNClassifier
import numpy as np
from mpi4py import MPI
import time

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start_time = MPI.Wtime()

    rows = 100000
    cols = 500
    
    # Emaulate a parallel reading of the dataset 
    np.random.seed(699)
    X_train = np.random.rand(rows * cols).reshape((rows, cols))
    y_train = np.random.randint(2, size=rows)
    # ---------------------------------------------------------

    knn = KNNClassifier(k=2)
    knn.fit(X_train, y_train)

    test_size = 1000    
    assert size <= test_size, "Use less cores to run the program."
    if rank == 0:
        print("---------------------------------------------------------------------")
        print("MPI KNN CLASSIFIER")
        print(f"Number of cores: \t{size}\n")
        print(f"Data set dimension:\t{rows}x{cols}")
        print(f"Number of tests:\t{test_size}")
        print("---------------------------------------------------------------------")
    X_test_indices = np.random.randint(rows, size=test_size)
    X_test = X_train[X_test_indices]

    if rank == 0:
        print(" Start prediction...")
        print(f'X_train shape {X_train.shape} - y_train shape {y_train.shape}')
    predictions = knn.predict(X_test)
    
    end_time = MPI.Wtime()

    if rank == 0:
        #print(f'Predictions: {predictions}')
        #print(f'Label:      {y_train[X_test_indices]}')
        correct = np.sum(y_train[X_test_indices] == predictions)
        print(f'Correct predictions: {correct}/{test_size}')
        print(f"Execution time: {end_time - start_time} s")
        with open("mpi_knn_results.csv", "a") as file:
            file.write(f"{size}, {rows}, {test_size}, {end_time - start_time}\n")
        
if __name__ == "__main__":
    main()

