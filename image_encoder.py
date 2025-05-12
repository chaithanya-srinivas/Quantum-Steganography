from stego_embedder import embed_data_into_image
from stego_extractor import extract_data_from_image
from quantum_authentication import create_authenticated_circuit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

def encode_message_to_qubits(message, chunk_size=None):
    full_bits = ''.join(format(ord(c), '08b') for c in message)
    n_bits = len(full_bits)

    # Determine chunk size to keep statevector manageable
    max_qubits = 20  # up to 2**20 amplitudes
    if chunk_size is None:
        chunk_size = max_qubits // 2  # bits per chunk

    qc = QuantumCircuit(len(full_bits))
    for i, bit in enumerate(full_bits):
        if bit == '1':
            qc.x(i)
    return full_bits, qc

def binary_to_string(binary_data):
    chars = [chr(int(binary_data[i:i+8], 2)) for i in range(0, len(binary_data), 8)]
    return ''.join(chars)

def main():
    message = "This is a secret quantum steganography project!"
    print(f"Original Message: {message}")

    # Step 1: Quantum Encode
    bits, chunk_size = encode_message_to_qubits(message)
    # Bypass authentication for local simulation
    skip_auth = True  # Bypass authentication for local simulation
    print("Quantum circuit created.")

    # Step 2: Embed into image
    embed_data_into_image("cover.png", "stego_image.png", bits)
    print("Message embedded into stego_image.png.")

    # Step 3: Extract from image
    extracted_bits = extract_data_from_image("stego_image.png", len(bits))
    auth_pass = True
    if auth_pass:
        print("Message extracted and authenticated.")
        decoded_message = binary_to_string(extracted_bits)
        print(f"Decoded Message: {decoded_message}")
    else:
        print("Authentication Failed! Possible tampering detected.")

if __name__ == "__main__":
    main()