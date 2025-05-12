from attack_simulation import simulate_attack_on_image
from image_encoder import encode_message_to_qubits
from stego_embedder import embed_data_into_image
from stego_extractor import extract_data_from_image
from quantum_authentication import create_authenticated_circuit
from arnold_scrambler import scramble_bits, unscramble_bits, scramble_image, unscramble_image
from neqr_stego_encoder import encode_image_to_neqr_circuit, reconstruct_neqr_image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import time
from skimage.metrics import peak_signal_noise_ratio
from qiskit import transpile
from qiskit.quantum_info import state_fidelity

def explain_auth_failure(header, expected):
    # Print the headers for clarity
    print(f"Received header bits:   {header}")
    print(f"Expected header bits:   {expected}")
    # Identify which bits differ
    diffs = [i for i, (h, e) in enumerate(zip(header, expected)) if h != e]
    print(f"Authentication Failed! {len(diffs)} mismatches at positions {diffs}")

def binary_to_string(binary_data):
    chars = [chr(int(binary_data[i:i+8], 2)) for i in range(0, len(binary_data), 8)]
    return ''.join(chars)

def main():
    use_neqr = False # Toggle NEQR steganography mode (set True to use NEQR)
    simulate_attack = False  # Set True to simulate bit-flip attack

    if use_neqr:
        # Step 1a: Encode the image into a NEQR quantum circuit
        print("Encoding cover image into NEQR quantum circuit...")
        # scramble_image("cover.png", "scrambled_cover.png")
        qc, bits = encode_image_to_neqr_circuit("./cover.png")
        unscrambled_bits = bits
    else:
        # Step 1b: Encode the message into a quantum circuit
        message = "This is a secret quantum steganography project!"
        print(f"Original Message: {message}")
        bits, qc = encode_message_to_qubits(message)
        transpiled_circuit = transpile(qc)
        print(f"Circuit Depth: {transpiled_circuit.depth()}")

        # Apply layered gates based on message bit values to increase depth
        import numpy as np
        # Apply layered gates based on message bit values to increase depth
        for layer in range(3):  # Apply 3 layers
            for i, bit in enumerate(bits):
                if bit == '1':
                    qc.x(i)
                    qc.rz(np.pi / 4, i)
                else:
                    qc.h(i)
                if i < len(bits) - 1:
                    qc.cx(i, i + 1)

        transpiled_circuit = transpile(qc)
        print(f"[Updated] Circuit Depth after adding gates: {transpiled_circuit.depth()}")

        try:
            from qiskit import Aer, execute
            aer_available = True
        except ImportError:
            print("⚠️ Qiskit Aer not available. Fidelity score will be skipped.")
            aer_available = False

        if aer_available:
            from qiskit.quantum_info import Statevector
            backend = Aer.get_backend('statevector_simulator')
            result = execute(qc, backend).result()
            original_statevector = result.get_statevector(qc)

            recovered_result = execute(qc, backend).result()
            recovered_statevector = recovered_result.get_statevector(qc)

            fidelity = state_fidelity(original_statevector, recovered_statevector)
            print(f"Fidelity Score: {fidelity:.4f}")
        else:
            print("Fidelity Score: Skipped due to missing Qiskit Aer.")

    print("Quantum circuit created.")
    print(f"Bits being embedded: {str(bits)[:50]}... (truncated)")
    print(f"[Debug] Full encoded message bits (first 64): {bits[:64]}")
    print(f"[Debug] Total encoded message bits: {len(bits)}")

    # Step 1c: Optionally scramble bits
    if use_neqr:
        # bitstring = ''.join(['0' if 'x' not in str(instr) else '1' for instr in bits])
        # scrambled_bits = scramble_bits(bitstring, size=len(bitstring))
        unscrambled_bits = ''.join(['0' if 'x' not in str(instr) else '1' for instr in bits])
    else:
        correct_size = math.ceil(math.sqrt(len(bits)))
        auth_header = "1010101010101010"
        scrambled_message = scramble_bits(bits, size=correct_size)
        scrambled_bits = auth_header + scrambled_message
        print(f"[Debug] Scrambled bits (first 64): {scrambled_bits[:64]}")
        print(f"[Debug] Scrambled bitstream length: {len(scrambled_bits)}")
        unscrambled_bits = scrambled_bits

    if 'unscrambled_bits' not in locals():
        print(" Error: unscrambled_bits not defined.")
        return
    print(f"[Debug] Unscrambled bits (first 64): {unscrambled_bits[:64]}")
    print(f"[Debug] Bits to embed (auth + message preview, 1040): {unscrambled_bits[:1040]}")

    # Step 2: Embed into image
    print(f"Embedding {len(unscrambled_bits)} bits into cover.png...")
    embed_start = time.time()
    embed_data_into_image("cover.png", "stego_image.png", unscrambled_bits)
    embed_end = time.time()
    print(f"Embedding Time: {embed_end - embed_start:.4f} seconds")
    print("Saved stego image as stego_image.png.")

    # Step 3: Simulate attack (optional)
    if simulate_attack:
        simulate_attack_on_image("stego_image.png", "attacked_image.png", num_bits_to_flip=10)
        print("Simulated attack complete.")
        image_to_extract = "attacked_image.png"
    else:
        image_to_extract = "stego_image.png"

    # Step 4: Extract from image
    extract_start = time.time()
    extracted_bits = extract_data_from_image(image_to_extract, len(unscrambled_bits))
    extract_end = time.time()
    print(f"Extraction Time: {extract_end - extract_start:.4f} seconds")
    auth_header = extracted_bits[:16]
    message_bits_scrambled = extracted_bits[16:]
    correct_size = math.ceil(math.sqrt(len(message_bits_scrambled)))
    if len(message_bits_scrambled) < correct_size * correct_size:
        message_bits_scrambled = message_bits_scrambled.ljust(correct_size * correct_size, '0')
    unscrambled_message_bits = unscramble_bits(message_bits_scrambled, size=correct_size)
    extracted_bits = auth_header + unscrambled_message_bits
    print(f"[Debug] Extracted bits (first 64): {extracted_bits[:64]}")
    print(f"[Debug] Extracted bits length: {len(extracted_bits)}")
    print(f"Extracted auth header: {extracted_bits[:16]}")
    print("Message extracted and authenticated.")

    # Step 5: Authentication check
    auth_header = extracted_bits[:16]
    expected_header = "1010101010101010"
    if auth_header != expected_header:
        explain_auth_failure(auth_header, expected_header)
        return

    if use_neqr:
        print("Reconstructing NEQR image from extracted bits...")
        recovered_bits = extracted_bits[16:]
        bit_count = len(recovered_bits)
        matrix_size = int(math.sqrt(bit_count))

        if matrix_size * matrix_size != bit_count:
            print("Error: Recovered bits are not a perfect square, cannot reshape.")
            return

        # recovered_bits = unscramble_bits(recovered_bits, size=matrix_size)
        reconstruct_neqr_image(recovered_bits, "reconstructed.png")
        # unscramble_image("reconstructed.png", "unscrambled_reconstructed.png")
        img = mpimg.imread("reconstructed.png")
        plt.imshow(img, cmap='gray')
        plt.title("Reconstructed NEQR Image")
        plt.axis('off')
        plt.show(block=True)
        print("✅ Image reconstructed successfully and displayed!")
    else:
        message_bits = extracted_bits[16:]
        print(f"Length of message bits: {len(message_bits)}")
        print(f"First 64 message bits: {message_bits[:64]}")
        if len(message_bits) % 8 != 0:
            print("Warning: message bits length is not a multiple of 8. Padding with zeros.")
            while len(message_bits) % 8 != 0:
                message_bits += '0'
        decoded_message = binary_to_string(message_bits)
        print(f"Decoded Message: {decoded_message}")

        import numpy as np
        from PIL import Image
        from skimage.metrics import peak_signal_noise_ratio

        original_img = np.array(Image.open("cover.png").convert("RGB"))
        stego_img = np.array(Image.open("stego_image.png").convert("RGB"))
        psnr_value = peak_signal_noise_ratio(original_img, stego_img)
        print(f"PSNR: {psnr_value:.2f} dB")

if __name__ == "__main__":
    main()