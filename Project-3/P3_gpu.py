import cupy as cp
import numpy as np
import torch
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

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

# Trasferisci i dati alla GPU
x_gpu = cp.array(x)
y_true_gpu = cp.array(y_true)

# Definisci la funzione di convoluzione utilizzando CuPy
def convolution_2d(x, kernel):
    input_height, input_width = x.shape
    kernel_height, kernel_width = kernel.shape
    pad_height, pad_width = kernel_height // 2, kernel_width // 2

    # Pad the input array by adding extra pixel
    padded_x = cp.pad(x, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Initialize the output matrix
    output_data = cp.zeros_like(x)

    # Perform the convolution operation
    for i in range(input_height):
        for j in range(input_width):
            # Extract the region of interest
            region = padded_x[i:i + kernel_height, j:j + kernel_width]
            # Perform element-wise multiplication and summation
            output_data[i, j] = cp.sum(region * kernel)

    return output_data

# Definisci la funzione di perdita
def loss_fn(kernel, x, y_true):
    y_pred = convolution_2d(x, kernel)
    return cp.mean((y_pred - y_true) ** 2)

# Inizializza il kernel
kernel = cp.array([[0.01, 0.0, 0.0],
                   [-1.0, 0.0, 1.0],
                   [0.0, 0.0, 0.0]], dtype=cp.float32)  # Random kernel for horizontal edge detection

# Trasferisci il kernel a PyTorch per il calcolo del gradiente
kernel_torch = torch.tensor(kernel.get(), requires_grad=True, device='cuda')

# Training loop
learning_rate = 0.01
num_iterations = 11

losses = []
for i in range(num_iterations):
    # Calcola la perdita
    loss = loss_fn(cp.array(kernel_torch.detach().cpu().numpy()), x_gpu, y_true_gpu)
    
    # Calcola il gradiente
    loss_torch = torch.tensor(loss.get(), requires_grad=True, device='cuda')
    loss_torch.backward()
    
    # Aggiorna il kernel
    with torch.no_grad():
        kernel_torch -= learning_rate * kernel_torch.grad
        kernel_torch.grad.zero_()
    
    # Memorizza la perdita
    losses.append(loss)

    # Stampa la perdita ogni 10 iterazioni
    if i % 10 == 0:
        print(f"Iteration {i}, Loss: {loss:.4f}")

# Visualizza i risultati
plt.figure(figsize=(8, 6))

# Plot loss over iterations
plt.subplot(2, 2, 1)
plt.plot(cp.asnumpy(cp.array(losses)))
plt.title("Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")
# plt.savefig('loss_curve.png')

# Display original noisy image
plt.subplot(2, 2, 2)
plt.imshow(cp.asnumpy(x_gpu), cmap='gray')
plt.title("Noisy Image")
plt.axis('off')
# plt.savefig('noisy_image.png')

# Display target clean image
plt.subplot(2, 2, 3)
plt.imshow(cp.asnumpy(y_true_gpu), cmap='gray')
plt.title("Target (Clean Image)")
plt.axis('off')
# plt.savefig('clean_image.png')

# Display denoised image
y_denoised = convolution_2d(x_gpu, cp.array(kernel_torch.detach().cpu().numpy()))
plt.subplot(2, 2, 4)
plt.imshow(cp.asnumpy(y_denoised), cmap='gray')
plt.title("Denoised Image")
plt.axis('off')
# plt.savefig('denoised_image.png')

plt.tight_layout()
plt.savefig('results.png')
plt.show()