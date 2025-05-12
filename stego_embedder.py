from PIL import Image
import numpy as np

def embed_data_into_image(input_image_path, output_image_path, binary_data):
    image = Image.open(input_image_path).convert("RGB")
    pixels = np.array(image)
    flat_pixels = pixels.reshape(-1).astype(np.uint8)

    full_data = binary_data
    print(f"Bits being embedded: {full_data[:20]}")

    # Check if there are enough pixels
    if len(full_data) > len(flat_pixels):
        raise ValueError("Cover image too small to embed the secret data!")

    print(f"Embedding {len(full_data)} bits into image...")

    for i in range(len(full_data)):
        flat_pixels[i] = (flat_pixels[i] & 254) | int(full_data[i])

    stego_pixels = flat_pixels.reshape(pixels.shape)
    stego_image = Image.fromarray(stego_pixels.astype(np.uint8), "RGB")
    stego_image.save(output_image_path)
    print(f"Saved stego image as {output_image_path}")

    # --- New Debugging Step: Compare the embedded bits ---
    extracted_bits = ''.join([str(stego_pixels.flatten()[i] & 1) for i in range(len(full_data))])

    print("\n--- Deep Debugging ---")
    print(f"Expected bits (first 20): {full_data[:20]}")
    print(f"Embedded bits (first 20): {extracted_bits[:20]}")
    if full_data[:20] == extracted_bits[:20]:
        print("Embedding verified for first 20 bits.")
    else:
        print("Warning: Mismatch detected in first 20 bits!")

    # --- Full image verification ---
    if full_data == extracted_bits:
        print("Full embedding verification successful: all bits match.")
    else:
        mismatches = [i for i in range(len(full_data)) if full_data[i] != extracted_bits[i]]
        print(f" Full embedding verification failed: {len(mismatches)} mismatches detected.")