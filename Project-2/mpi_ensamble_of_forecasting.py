from mpi4py import MPI
import jax
import jax.numpy as jnp
import numpy as np

def forecast_1step(X:jnp.array, W:jnp.array, b:jnp.array)->jnp.array:
    X_flatten = X.flatten()
    y_next = jnp.dot(W, X_flatten) + b
    return y_next

def forecast(horizon:int, X:jnp.array, W:jnp.array, b:jnp.array)->jnp.array:
    result = []

    for t in range(horizon):
        X_flatten = X.flatten()
        y_next = forecast_1step(X_flatten, W, b)
        X = jnp.roll(X, shift=-1, axis=0)
        X = X.at[-1].set(y_next)
        result.append(y_next)

    return jnp.array(result)

def forecast_1step_with_loss(params:tuple, X:jnp.array, y:jnp.array)->float:
    W, b = params
    y_next = forecast_1step(X, W, b)
    return jnp.sum((y_next - y) ** 2)

grad = jax.grad(forecast_1step_with_loss)

def training_loop(grad:callable, num_epochs:int, W:jnp.array, b:jnp.array, X:jnp.array, y:jnp.array)->tuple:
    for i in range(num_epochs):
        delta = grad((W, b), X, y)
        W -= 0.1 * delta[0]
        b -= 0.1 * delta[1]
    return W, b

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("MPI size:", size)
        print("Creation of dataset and model parameters")
        X = jnp.array([[0.1, 0.4], [0.1, 0.5], [0.1, 0.6]])
        y = jnp.array([[0.1, 0.7]])
        W = jnp.array([[0., 1., 0., 1., 0., 1.], [0., 1., 0, 1., 0., 1.]])
        b = jnp.array([0.1])

    if rank == 0:
        data = (X, y, W, b)
    else:
        data = None

    data = comm.bcast(data, root=0)
    X, y, W, b = data

    num_epochs = 1000
    horizon = 10
    num_forecaster = 10
    noise_std = 0.1

    forecast_per_processor = num_forecaster // size
    residual_forecaster = num_forecaster % size
    if rank == 0:
        counts = [forecast_per_processor * horizon * X.shape[1]for _ in range(size)]
        counts[0] += residual_forecaster * horizon * X.shape[1]
        displacements = np.cumsum([0] + counts[:-1])
    else:
        counts = None
        displacements = None

    partial_prediction = []

    for forecaster in range(forecast_per_processor + (1 if rank == 0 and residual_forecaster > 0 else 0)):
        key = jax.random.PRNGKey(rank)
        W_noise = jax.random.normal(key, W.shape) * noise_std
        b_noise = jax.random.normal(key, b.shape) * noise_std

        W_init = W + W_noise
        b_init = b + b_noise

        W_trained, b_trained = training_loop(grad, 20, W_init, b_init, X, y)
        y_predicted_local = forecast(horizon, X, W_trained, b_trained)
        partial_prediction.append(y_predicted_local.flatten())

    partial_prediction_np = np.array(partial_prediction)

    if rank == 0:
        y_pred = np.empty((num_forecaster * horizon * X.shape[1],), dtype=np.float32)
    else:
        y_pred = None
    
    comm.Gatherv(sendbuf=partial_prediction_np, recvbuf=(y_pred, counts, displacements, MPI.FLOAT), root=0)

    if rank == 0:
        # y_pred = y_pred.reshape((num_forecaster, horizon))
        # print("Forecasted values:", y_pred)
        print("size of forecasted values:", y_pred.shape)

if __name__ == "__main__":
    main()