from arnold_scrambler import scramble_image, unscramble_image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def resize_image_to_square(path, size=32):
    img = Image.open(path).convert('L')
    width, height = img.size
    min_edge = min(width, height)
    left = (width - min_edge) // 2
    top = (height - min_edge) // 2
    right = left + min_edge
    bottom = top + min_edge
    img = img.crop((left, top, right, bottom))  # Crop to square
    img = img.resize((size, size))
    img.save(path)

# File paths
original = "cover.png"
scrambled = "scrambled_test.png"
unscrambled = "unscrambled_test.png"

# Number of iterations
iterations = 5

resize_image_to_square(original, size=256)
# Step 1: Scramble
scramble_image(original, scrambled, iterations)

# Step 2: Unscramble
unscramble_image(scrambled, unscrambled, iterations)

# Step 3: Display results
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
for ax, img_path, title in zip(axs, [original, scrambled, unscrambled], ["Original", "Scrambled", "Unscrambled"]):
    img = Image.open(img_path).convert('L')
    ax.imshow(img, cmap='gray')
    # PSNR calculation for Scrambled image
    from skimage.metrics import peak_signal_noise_ratio as psnr
    if title == "Scrambled":
        psnr_value = psnr(np.array(Image.open(original).convert('L')),
                          np.array(Image.open(scrambled).convert('L')))
        print(f"[Image] PSNR between Original and Scrambled: {psnr_value:.2f} dB")
    # PSNR calculation for Unscrambled image
    if title == "Unscrambled":
        psnr_value_unscrambled = psnr(np.array(Image.open(original).convert('L')),
                                     np.array(Image.open(unscrambled).convert('L')))
        print(f"[Image] PSNR between Original and Unscrambled: {psnr_value_unscrambled:.2f} dB")
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()

# Quantum circuit visualization of scrambled image bits
from qiskit import QuantumCircuit, transpile
import numpy as np
scrambled_img = Image.open(scrambled).convert('L')
scrambled_array = np.array(scrambled_img)
scrambled_bits = ''.join(format(pixel, '08b') for pixel in scrambled_array.flatten())
scrambled_bits = scrambled_bits[:64]  # limit to 64 bits for visualization

# Create quantum circuit with Pauli-X gates for '1' bits
qc = QuantumCircuit(len(scrambled_bits), name="ScramblerCircuit")
for i, bit in enumerate(scrambled_bits):
    if bit == '1':
        qc.x(i)

# Add Hadamard and some entanglement to increase depth
for i in range(len(scrambled_bits)):
    qc.h(i)
    if i < len(scrambled_bits) - 1:
        qc.cx(i, i + 1)

# Transpile and report circuit depth
transpiled_circuit = transpile(qc)
print(f"[Quantum] Scrambler Circuit Depth: {transpiled_circuit.depth()}")
print(qc.draw(output='text'))