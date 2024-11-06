from mpi4py import MPI
import numpy as np
import math

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        diff = (x1 - x2)
        return np.sqrt(np.sum(diff ** 2))

    def predict(self, X):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        samples = X.shape[0]
        chunks = samples // size
        extra_samples = samples - chunks*(size - 1)  # extra samples dim

        if extra_samples == 0:
            local_X = np.array_split(X, size)[rank]
            local_y_pred = np.array([self._predict(x) for x in local_X], dtype=int)
            y_pred = None
            if rank == 0:
                y_pred = np.empty(samples, dtype=int)
            comm.Gather(local_y_pred, y_pred, root=0)
            return y_pred if rank == 0 else None
        else:
            if rank == size - 1:
                local_samples = extra_samples
            else:
                local_samples = chunks  

            start_index = rank * chunks 
            end_index = start_index + local_samples

            local_X = X[start_index:end_index]
            local_y_pred = np.array([self._predict(x) for x in local_X], dtype=int)

            counts = [chunks if i < size - 1 else extra_samples for i in range(size)] 
            if rank == 0:
                y_pred = np.empty(samples, dtype=int)
            else:
                y_pred = None
                counts = None
            comm.Gatherv(sendbuf=local_y_pred, recvbuf=(y_pred, counts), root=0)
            return y_pred if rank == 0 else None

    def _predict(self, x):
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common



