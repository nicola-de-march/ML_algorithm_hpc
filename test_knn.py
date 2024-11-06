from KNNClassifier import KNNClassifier
import numpy as np
from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    rows = 1000
    cols = 500
    np.random.seed(699)
    X_train = np.random.rand(rows * cols).reshape((rows, cols))
    y_train = np.random.randint(2, size=rows)

    knn = KNNClassifier(k=2)
    knn.fit(X_train, y_train)

    test_size = 100
    X_test_indices = np.random.randint(rows, size=test_size)
    X_test = X_train[X_test_indices]

    if rank == 0:
        print(f'X_train shape {X_train.shape} - y_train shape {y_train.shape}')
    predictions = knn.predict(X_test)
    
    if rank == 0:
        #print(f'Predictions: {predictions}')
        #print(f'Label:      {y_train[X_test_indices]}')
        print(len(predictions))
        correct = np.sum(y_train[X_test_indices] == predictions)
        print(f'Correct predictions: {correct}/{test_size}')

if __name__ == "__main__":
    main()
