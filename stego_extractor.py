from PIL import Image
import numpy as np

def extract_data_from_image(image_path, num_bits):
    image = Image.open(image_path)
    image = image.convert("RGB")
    pixels = np.array(image)
    flat_pixels = pixels.flatten()

    bits = [str(flat_pixels[i] & 1) for i in range(num_bits)]
    return ''.join(bits)