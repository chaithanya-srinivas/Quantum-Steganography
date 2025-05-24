import socket

HOST = '127.0.0.1'
PORT = 5005
MESSAGE = "Transaction ID: CS2-XYZ987"

# Send message to server
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))
client_socket.sendall(MESSAGE.encode())
print("[Client] Sent message to server.")

# Receive stego image bytes
image_data = b""
while True:
    chunk = client_socket.recv(4096)
    if not chunk:
        break
    image_data += chunk

# Save received image
image_path = "received_stego_image.png"
with open(image_path, 'wb') as f:
    f.write(image_data)

print(f"[Client] Stego image saved as {image_path}")
client_socket.close()