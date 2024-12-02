import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import jax
import jax.numpy as jnp
from jax import grad
from mpi4py import MPI
import sys

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

# JIT-compiled convolution function
convolution_2d_jit = jax.jit(convolution_2d)

# Define loss function
def loss_fn(kernel, x, y_true):
    y_pred = convolution_2d_jit(x, kernel)
    return jnp.mean((y_pred - y_true) ** 2)  # Mean squared error

# JAX JIT-compiled loss function
loss_fn_jit = jax.jit(loss_fn)

# -------------------------------------------------------------------------------

(x_train, y_train), (x_test, y_test) = mnist.load_data()
if(len(sys.argv) < 2):
    error("Usage: python3 P3.py <N_IMAGES>")

N_IMAGES = int(sys.argv[1])

chunck_size = int(N_IMAGES / size)
if rank == 0:
    print("chunck_size:  " + str(chunck_size))
x_local = x_train[(rank*chunck_size) : (rank*chunck_size + chunck_size)]

y_true_local = x_local.copy()

# Add salt-and-pepper noise
num_corrupted_pixels = 100
rand_i = x_local[0].shape[0]; rand_j = x_local[0].shape[1]
for x in x_local:
    for _ in range(num_corrupted_pixels):
        i, j = np.random.randint(0, rand_i), np.random.randint(0, rand_j)
        x[i, j] = np.random.choice([0, 255])

# Normalize images
y_true_local = y_true_local.astype(np.float32) / 255.0
x_local = x_local.astype(np.float32) / 255.0

kernel = jnp.array([[0.01, 0.0, 0.0],
                    [-1.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0]])  # Random kernel for horizontal edge detection

# Gradient of the loss function w.r.t. the kernel
loss_grad = grad(loss_fn_jit)
loss_grad_jit = jax.jit(loss_grad)

# Training loop
learning_rate = 0.01
num_iterations = 200

if rank == 0:
    print("image shape: ", x_local.shape)
    print("kernel shape: ", kernel.shape)
    print("learning_rate: ", learning_rate)
    print("num_iterations: ", num_iterations)
    print("Training start -----------------------------------------------------------------")

sys.stdout.flush()
comm.Barrier()
losses = []

time_iter = 0
tot_time_start = MPI.Wtime()

for x, y in zip(x_local, y_true_local):
    for i in range(num_iterations):
        start_time = MPI.Wtime()
        gradients_local = loss_grad_jit(kernel, x, y)
        gradients_local = np.array(gradients_local)
        gradients = np.zeros_like(gradients_local)
        comm.Allreduce(gradients_local, gradients, op=MPI.SUM)
        # Average gradients
        gradients_avg = gradients / size
        # Update kernel with averaged gradients
        kernel -= learning_rate * gradients_avg
        # Compute and store the loss
        current_loss = loss_fn(kernel, x, y)
        losses.append(current_loss)
        # Print loss every 10 iterations
        end_time = MPI.Wtime()
        print(f"Iteration {i}, Loss rank {rank}: {current_loss:.4f}, Time per iteration: {end_time - start_time:.4f}")
        time_iter += end_time - start_time
tot_time = MPI.Wtime() - tot_time_start

sys.stdout.flush()
comm.Barrier()

if rank == 0:
    print("Training end -----------------------------------------------------------------")
    print("Final kernel: ", kernel)
    print("Final loss: ", losses[-1])
    print(f"Total training time: {tot_time} [s]")
    print("\nSaving results...")
    f = open("time_analysis_jax.csv", "a")
    f.write(f"mpi_jax, {size}, {N_IMAGES}, {num_iterations}, {tot_time}, {time_iter/(num_iterations)}\n")
    f.close()
# # Visualize results
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
# plt.imshow(x_local, cmap='gray')
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
# y_denoised = convolution_2d(x_local, kernel)
# plt.subplot(2, 2, 4)
# plt.imshow(y_denoised, cmap='gray')
# plt.title("Denoised Image")
# plt.axis('off')
# # plt.savefig('denoised_image.png')

# plt.tight_layout()
# plt.savefig(f'results/results_{rank}.png')
# plt.show()

sys.stdout.flush()
comm.Barrier()
if rank == 0:
    print("Results saved successfully!")


