# neqr_stego_encoder.py

# Author: Chaithanya S
# Date: 2025-04-30
# Purpose: Simulate NEQR (Novel Enhanced Quantum Representation) and basic Quantum Steganography Encoding
# Note: For Grayscale images only (8-bit, pixel values from 0 to 255)
# The encode_image_to_neqr_circuit function returns (bitstring, qc)

from qiskit import QuantumCircuit
import numpy as np
from PIL import Image
from arnold_scrambler import scramble_bits, unscramble_bits

def load_grayscale_image(image_path):
    """Loads a grayscale image and returns pixel values as a numpy array."""
    image = Image.open(image_path).convert('L')  # 'L' mode = grayscale
    pixels = np.array(image)
    return pixels

def create_neqr_circuit(pixels):
    """Creates a NEQR quantum circuit for the given grayscale pixel array."""
    n = int(np.log2(pixels.shape[0]))  # Assume square images (2^n x 2^n)
    q = 8  # 8 bits for grayscale intensity

    qc = QuantumCircuit(n + q)

    # Encode pixel positions and values into the quantum circuit
    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            pixel_value = pixels[i, j]
            position = i * pixels.shape[1] + j
            position_bits = format(position, f'0{n*2}b')
            pixel_bits = format(pixel_value, '08b')

            # Apply X gates for position bits
            for idx, bit in enumerate(position_bits):
                if bit == '1' and idx < qc.num_qubits:
                    qc.x(idx)

            # Apply controlled X gates for pixel bits
            for idx, bit in enumerate(pixel_bits):
                control_idx = idx
                target_idx = n + idx
                if bit == '1' and control_idx < qc.num_qubits and target_idx < qc.num_qubits:
                    qc.cx(control_idx, target_idx)

            # Reset position bits back
            for idx, bit in enumerate(position_bits):
                if bit == '1' and idx < qc.num_qubits:
                    qc.x(idx)

    return qc

def save_circuit(qc, output_path):
    """Saves the quantum circuit diagram to an image."""
    from qiskit.visualization import circuit_drawer
    circuit_drawer(qc, output=output_path, scale=0.8)

def encode_image_to_neqr_circuit(image_path):
    """Interface function to load a grayscale image, generate a NEQR circuit, and extract classical bits.
    Returns (bitstring, qc).
    """
    pixels = load_grayscale_image(image_path)
    qc = create_neqr_circuit(pixels)

    # Simulate the circuit to extract classical bits
    from qiskit.quantum_info import Statevector

    state = Statevector.from_instruction(qc)
    probabilities = state.probabilities_dict()

    # Find the most likely outcome
    max_key = max(probabilities, key=probabilities.get)
    bitstring = max_key[::-1]  # Reverse order to match LSB embedding

    scrambled_bitstring = scramble_bits(bitstring, size=len(bitstring))
    return scrambled_bitstring, qc

def main():
    image_path = 'cover.png'  # Your input image
    output_circuit_image = 'neqr_circuit.png'

    pixels = load_grayscale_image(image_path)
    qc = create_neqr_circuit(pixels)
    save_circuit(qc, output_circuit_image)
    print(f"NEQR Circuit generated and saved as {output_circuit_image}")

if __name__ == "__main__":
    main()


# Extraction placeholder function

def extract_bits_from_neqr_circuit(bits):
    """Unscrambles NEQR bitstring before passing to reconstruction."""
    unscrambled = unscramble_bits(bits, size=len(bits))
    return unscrambled


# New function: reconstruct_neqr_image
def reconstruct_neqr_image(bits, save_path=None):
    """Reconstructs a grayscale square image from a bitstring."""
    if len(bits) % 8 != 0 or not np.sqrt(len(bits) // 8).is_integer():
        print("Bitstring length is not a valid square image.")
        return None

    pixels = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        pixel = int(byte, 2)
        pixels.append(pixel)

    side = int(np.sqrt(len(bits) // 8))
    image_array = np.array(pixels, dtype=np.uint8).reshape((side, side))
    if save_path:
        image = Image.fromarray(image_array)
        image.save(save_path)
        print(f"Reconstructed image saved to {save_path}")
    return image_array
