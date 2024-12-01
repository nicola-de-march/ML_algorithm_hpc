import numpy as np
# import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import jax
import jax.numpy as jnp
from jax import grad
import mpi4py as MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define convolution function using JAX
def convolution_2d(x, kernel):
    input_height, input_width = x.shape
    kernel_height, kernel_width = kernel.shape
    pad_height, pad_width = kernel_height // 2, kernel_width // 2

    # Pad the input array by adding extra pixel
    padded_x = jnp.pad(x, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Initialize the output matrix
    output_data = jnp.zeros_like(x)

    # Perform the convolution operation
    for i in range(input_height):
        for j in range(input_width):
            # Extract the region of interest
            region = padded_x[i:i + kernel_height, j:j + kernel_width]
            # Perform element-wise multiplication and summation
            output_data = output_data.at[i, j].set(jnp.sum(region * kernel))
            # Equivalent to : output_data[i, j] = jnp.sum(region * kernel)

    return output_data

# Define loss function
def loss_fn(kernel, x, y_true):
    y_pred = convolution_2d(x, kernel)
    return jnp.mean((y_pred - y_true) ** 2)  # Mean squared error
# -------------------------------------------------------------------------------

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_local = x_train[rank]
y_true_local = x_local.copy()

# Add salt-and-pepper noise
num_corrupted_pixels = 100
for _ in range(num_corrupted_pixels):
    i, j = np.random.randint(0, x_local.shape[0]), np.random.randint(0, x_local.shape[1])
    x_local[i, j] = np.random.choice([0, 255])

# Normalize images
y_true_local = y_true_local.astype(np.float32) / 255.0
x = x_local.astype(np.float32) / 255.0

kernel = jnp.array([[0.01, 0.0, 0.0],
                    [-1.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0]])  # Random kernel for horizontal edge detection

# Gradient of the loss function w.r.t. the kernel
loss_grad = grad(loss_fn)

# Training loop
learning_rate = 0.05
num_iterations = 200

losses = []
for i in range(num_iterations):
    gradients_local = loss_grad(kernel, x, y_true_local)
    # Send gradient to rank 0
    gradients_local = np.array(gradients_local)
    gradients = comm.reduce(gradients_local, op=MPI.SUM, root=0)
    
    if rank == 0:
        kernel -= learning_rate * gradients

    kernel = comm.bcast(kernel, root=0)
    kernel = jnp.array(kernel)
    # send kernel to the different process
    # Compute and store the loss
    current_loss = loss_fn(kernel, x_local, y_true_local)
    losses.append(current_loss)

    # Print loss every 10 iterations
    if rank == 0:
        print(f"Iteration {i}, Local Loss rank 0: {current_loss:.4f}")

if rank == 0:
    # Visualize results
    # plt.figure(figsize=(8, 6))

    # # Plot loss over iterations
    # plt.subplot(2, 2, 1)
    # plt.plot(losses)
    # plt.title("Loss Curve")
    # plt.xlabel("Iteration")
    # plt.ylabel("Loss")
    # # plt.savefig('loss_curve.png')

    # # Display original noisy image
    # plt.subplot(2, 2, 2)
    # plt.imshow(x, cmap='gray')
    # plt.title("Noisy Image")
    # plt.axis('off')
    # # plt.savefig('noisy_image.png')

    # # Display target clean image
    # plt.subplot(2, 2, 3)
    # plt.imshow(y_true_local, cmap='gray')
    # plt.title("Target (Clean Image)")
    # plt.axis('off')
    # # plt.savefig('clean_image.png')

    # # Display denoised image
    # y_denoised = convolution_2d(x, kernel)
    # plt.subplot(2, 2, 4)
    # plt.imshow(y_denoised, cmap='gray')
    # plt.title("Denoised Image")
    # plt.axis('off')
    # # plt.savefig('denoised_image.png')

    # plt.tight_layout()
    # plt.savefig('results.png')
    # plt.show()

    print("Training end")




