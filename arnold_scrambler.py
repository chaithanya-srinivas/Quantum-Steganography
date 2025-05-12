import numpy as np

def arnold_cat_map(image, iterations=1):
    """Applies the Arnold Cat Map to a square 2D numpy array."""
    N, M = image.shape
    if N != M:
        raise ValueError("Arnold Cat Map requires a square image (NxN).")
    scrambled = np.copy(image)
    for _ in range(iterations):
        temp = np.zeros_like(scrambled)
        for x in range(N):
            for y in range(N):
                vec = np.dot(np.array([[1, 1], [1, 2]]), np.array([x, y])) % N
                new_x, new_y = vec
                temp[new_x, new_y] = scrambled[x, y]
        scrambled = temp
    return scrambled

def inverse_arnold_cat_map(image, iterations=1):
    """Reverses the Arnold Cat Map on a square 2D numpy array."""
    N, M = image.shape
    if N != M:
        raise ValueError("Inverse Arnold Cat Map requires a square image (NxN).")
    unscrambled = np.copy(image)
    inverse_matrix = np.array([[2, -1], [-1, 1]])
    for _ in range(iterations):
        temp = np.zeros_like(unscrambled)
        for x in range(N):
            for y in range(N):
                vec = np.dot(inverse_matrix, np.array([x, y])) % N
                new_x, new_y = vec
                temp[new_x, new_y] = unscrambled[x, y]
        unscrambled = temp
    return unscrambled

def scramble_bits(bitstring, size, iterations=5):
    """Convert bitstring into square matrix, apply scrambling, then return string."""
    import numpy as np

    if isinstance(bitstring, str):
        bits = np.array([int(b) for b in bitstring])
    else:
        bits = np.array(bitstring)

    padded_len = size * size
    if len(bits) < padded_len:
        bits = np.pad(bits, (0, padded_len - len(bits)), constant_values=0)

    matrix = bits.reshape((size, size)).astype(np.uint8)

    def arnold_cat_map_binary(mat, iterations):
        n = mat.shape[0]
        for _ in range(iterations):
            new_mat = np.zeros_like(mat)
            for x in range(n):
                for y in range(n):
                    new_x = (x + y) % n
                    new_y = (x + 2 * y) % n
                    new_mat[new_x, new_y] = mat[x, y]
            mat = new_mat
        return mat

    scrambled_matrix = arnold_cat_map_binary(matrix, iterations)
    result = ''.join(str(b) for b in scrambled_matrix.flatten())
    print(f"Scrambling complete with {iterations} iterations.")
    return result

def unscramble_bits(bitstring, size, iterations=5):
    """Reverse scrambling on bitstring assumed to be square matrix."""
    if isinstance(bitstring, str):
        bits = np.array([int(b) for b in bitstring])
    else:
        bits = np.array(bitstring)
    matrix = bits.reshape((size, size))
    unscrambled_matrix = inverse_arnold_cat_map(matrix, iterations=iterations)
    result = ''.join(str(b) for b in unscrambled_matrix.flatten())
    print(f"Unscrambling complete with {iterations} iterations.")
    return result

# Additional functions for scrambling and unscrambling images
from PIL import Image

def scramble_image(image_path, output_path, iterations=1):
    """Scramble an image using the Arnold Cat Map and save the result."""
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    width, height = image.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    image = image.crop((left, top, right, bottom))
    image_array = np.array(image)
    scrambled_array = arnold_cat_map(image_array, iterations)
    scrambled_image = Image.fromarray(scrambled_array)
    scrambled_image.save(output_path)
    print(f"Scrambled image saved to {output_path} after {iterations} iterations.")

def unscramble_image(image_path, output_path, iterations=1):
    """Unscramble an image using the inverse Arnold Cat Map and save the result."""
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    width, height = image.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    image = image.crop((left, top, right, bottom))
    image_array = np.array(image)
    unscrambled_array = inverse_arnold_cat_map(image_array, iterations)
    unscrambled_image = Image.fromarray(unscrambled_array)
    unscrambled_image.save(output_path)
    print(f"Unscrambled image saved to {output_path} after {iterations} iterations.")