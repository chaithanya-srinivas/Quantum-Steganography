import numpy as np
from PIL import Image
import random

def simulate_attack_on_image(image_path, attacked_image_path, num_bits_to_flip=10):
    """
    Simulate an attack by randomly flipping a few least significant bits in the image.
    """
    image = Image.open(image_path).convert("RGB")
    pixels = np.array(image)
    flat_pixels = pixels.flatten()

    indices_to_flip = random.sample(range(len(flat_pixels)), num_bits_to_flip)
    for idx in indices_to_flip:
        flat_pixels[idx] ^= 1  # Flip the least significant bit

    attacked_pixels = flat_pixels.reshape(pixels.shape)
    attacked_image = Image.fromarray(attacked_pixels, "RGB")
    attacked_image.save(attacked_image_path)
    print(f"Simulated attack: {num_bits_to_flip} bits flipped, saved as {attacked_image_path}")