import socket
import io
from PIL import Image

HOST = '127.0.0.1'
PORT = 5005

def start_server():
    print("[Server] Starting socket server...")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)

    while True:
        conn, addr = server_socket.accept()
        print(f"[Server] Connection from {addr}")

        data = conn.recv(4096).decode()
        print(f"[Server] Received message: {data}")

        # Run stego pipeline (embed message into image)
        output_path = generate_stego_image_from_text(data)

        # Send image as bytes
        with open(output_path, 'rb') as img_file:
            img_bytes = img_file.read()

        conn.sendall(img_bytes)
        print("[Server] Sent stego image.\n")
        conn.close()

if __name__ == "__main__":
    start_server()

import time
from image_encoder import encode_message_to_qubits
from arnold_scrambler import scramble_bits
from stego_embedder import embed_data_into_image

def generate_stego_image_from_text(message):
    # Step 1: Quantum encode the message
    bits, _ = encode_message_to_qubits(message)

    # Step 2: Scramble the bits
    scrambled_bits = scramble_bits(bits, size=None, iterations=5)

    # Step 3: Add authentication header
    auth_header = '1010101010101010'
    full_bitstream = auth_header + scrambled_bits

    # Step 4: Embed into image
    timestamp = int(time.time())
    output_path = f"stego_output_{timestamp}.png"
    embed_data_into_image('cover.png', output_path, full_bitstream)
    return output_path