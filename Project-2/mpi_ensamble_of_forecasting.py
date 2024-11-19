from mpi4py import MPI
import jax # For installation: pip3 install jax, pip3 install jaxlib
import jax.numpy as jnp

def forecast_1step(X:jnp.array, W:jnp.array, b:jnp.array)->jnp.array:
    # JAX does not support in-place operations like numpy, so use jax.numpy and functional updates.
    # X = X.copy()  # Copy the input data to avoid modifying the original data
    X_flatten = X.flatten()
    y_next = jnp.dot(W, X_flatten) + b
    return y_next

def forecast(horizon:int, X:jnp.array, W:jnp.array, b:jnp.array)->jnp.array:
    result = []

    # Loop over 'horizon' to predict future values
    for t in range(horizon):
        X_flatten = X.flatten()  # Flatten the window for dot product

        # Get the next prediction
        y_next = forecast_1step(X_flatten, W, b)

        # Update X by shifting rows and adding the new prediction in the last row
        X = jnp.roll(X, shift=-1, axis=0)  # Shift rows to the left
        X = X.at[-1].set(y_next)  # Update the last row with the new prediction

        # Append the prediction to results
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
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("MPI size:", size)
        print("Creation of dataset and model parameters")
        X = jnp.array([[0.1, 0.4], [0.1, 0.5], [0.1, 0.6]])  # input example
        y = jnp.array([[0.1, 0.7]])  # expected output
        W = jnp.array([[0., 1., 0., 1., 0., 1.], [0., 1., 0, 1., 0., 1.]])  # random neural network parameters
        b = jnp.array([0.1])  # random neural network bias

    # Scatter the data to all processes
    if rank == 0:
        data = (X, y, W, b)
    else:
        data = None

    data = comm.bcast(data, root=0)

    X, y, W, b = data

    # Define the parameters
    num_epochs = 1000
    horizon = 10
    num_forecaster = 2 # <---- TODO: scale the number as you wish
    noise_std = 0.1 # the training needs to have different initial conditions for producing different predictions

    # Each process will have its own forecaster
    key = jax.random.PRNGKey(rank)  # Use rank as the random seed
    W_noise = jax.random.normal(key, W.shape) * noise_std
    b_noise = jax.random.normal(key, b.shape) * noise_std

    W_init = W + W_noise
    b_init = b + b_noise

    # Train the model
    W_trained, b_trained = training_loop(grad, 20, W_init, b_init, X, y)
    y_predicted = forecast(horizon, X, W_trained, b_trained)

    # Gather the predictions from all processes
    aggregated_forecasting = comm.gather(y_predicted, root=0)

    if rank == 0:
      print("Forecasted values:", aggregated_forecasting)

if __name__ == "__main__":
    main()