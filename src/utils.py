import numpy as np

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