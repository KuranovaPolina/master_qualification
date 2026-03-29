import numpy as np

import cv2

def get_projection_matrix(filename, matrix_name):
    if matrix_name not in ['P0', 'P1', 'P2', 'P3']:
        raise ValueError("Invalid name")

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith(f"{matrix_name}:"):
                values = line.split(':')[1].strip().split()
                data = np.array([float(v) for v in values])

                return data.reshape(3, 4)
    
    raise ValueError(f"Matrix not found")

def np2Img(np_image, Normalize=True):
    np_image = np.moveaxis(np_image, 0, -1)
    if Normalize:
        normalized = (np_image - np_image.min()) / (
            np_image.max() - np_image.min()) * 255.0
    else:
        normalized = np_image
    normalized = normalized[:, :, [2, 1, 0]]
    normalized = normalized.astype(np.uint8)
    return normalized


def np2Depth(input_tensor, invaild_mask):
    normalized = (input_tensor - input_tensor.min()) / (input_tensor.max() - input_tensor.min()) * 255.0
    normalized = normalized.astype(np.uint8)
    normalized = cv2.applyColorMap(normalized, cv2.COLORMAP_RAINBOW)
    normalized[invaild_mask] = 0
    return normalized