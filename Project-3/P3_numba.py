
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from numba import cuda, float32

# Carica il dataset MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = x_train[0]
y_true = x.copy()

# Aggiungi rumore sale e pepe
num_corrupted_pixels = 100
for _ in range(num_corrupted_pixels):
    i, j = np.random.randint(0, x.shape[0]), np.random.randint(0, x.shape[1])
    x[i, j] = np.random.choice([0, 255])

# Normalizza le immagini
y_true = y_true.astype(np.float32) / 255.0
x = x.astype(np.float32) / 255.0

# Definisci la funzione di convoluzione utilizzando Numba per GPU
@cuda.jit
def convolution_2d_gpu(x, kernel, output):
    input_height, input_width = x.shape
    kernel_height, kernel_width = kernel.shape
    pad_height, pad_width = kernel_height // 2, kernel_width // 2

    # Calcola le coordinate dell'elemento
    i, j = cuda.grid(2)
    if i < input_height and j < input_width:
        # Inizializza la somma
        sum = 0.0
        for ki in range(kernel_height):
            for kj in range(kernel_width):
                ni = i + ki - pad_height
                nj = j + kj - pad_width
                if 0 <= ni < input_height and 0 <= nj < input_width:
                    sum += x[ni, nj] * kernel[ki, kj]
        output[i, j] = sum

# Definisci la funzione di perdita
def loss_fn(kernel, x, y_true):
    output = np.zeros_like(x)
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(x.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(x.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    convolution_2d_gpu[blockspergrid, threadsperblock](x, kernel, output)
    return np.mean((output - y_true) ** 2)

# Inizializza il kernel
kernel = np.array([[0.01, 0.0, 0.0],
                   [-1.0, 0.0, 1.0],
                   [0.0, 0.0, 0.0]], dtype=np.float32)  # Random kernel for horizontal edge detection

# Training loop
learning_rate = 0.01
num_iterations = 100

losses = []
for i in range(num_iterations):
    # Calcola la perdita
    current_loss = loss_fn(kernel, x, y_true)
    losses.append(current_loss)

    # Calcola il gradiente numericamente
    grad_kernel = np.zeros_like(kernel)
    epsilon = 1e-4
    for ki in range(kernel.shape[0]):
        for kj in range(kernel.shape[1]):
            kernel[ki, kj] += epsilon
            loss_plus = loss_fn(kernel, x, y_true)
            kernel[ki, kj] -= 2 * epsilon
            loss_minus = loss_fn(kernel, x, y_true)
            kernel[ki, kj] += epsilon
            grad_kernel[ki, kj] = (loss_plus - loss_minus) / (2 * epsilon)

    # Aggiorna il kernel
    kernel -= learning_rate * grad_kernel

    # Stampa la perdita ogni 10 iterazioni
    if i % 10 == 0:
        print(f"Iteration {i}, Loss: {current_loss:.4f}")

# Visualizza i risultati
plt.figure(figsize=(8, 6))

# Plot loss over iterations
plt.subplot(2, 2, 1)
plt.plot(losses)
plt.title("Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")
# plt.savefig('loss_curve.png')

# Display original noisy image
plt.subplot(2, 2, 2)
plt.imshow(x, cmap='gray')
plt.title("Noisy Image")
plt.axis('off')
# plt.savefig('noisy_image.png')

# Display target clean image
plt.subplot(2, 2, 3)
plt.imshow(y_true, cmap='gray')
plt.title("Target (Clean Image)")
plt.axis('off')
# plt.savefig('clean_image.png')

# Display denoised image
output = np.zeros_like(x)
threadsperblock = (16, 16)
blockspergrid_x = int(np.ceil(x.shape[0] / threadsperblock[0]))
blockspergrid_y = int(np.ceil(x.shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)
convolution_2d_gpu[blockspergrid, threadsperblock](x, kernel, output)
plt.subplot(2, 2, 4)
plt.imshow(output, cmap='gray')
plt.title("Denoised Image")
plt.axis('off')
# plt.savefig('denoised_image.png')

plt.tight_layout()
plt.savefig('results.png')
plt.show()