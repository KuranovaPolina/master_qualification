from ultralytics import YOLO
import numpy as np
import cv2

from detect import detect_and_save
from distance_by_size import DistanceBySize

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

if __name__ == "__main__":
    model = YOLO('model/yolo26n.pt')
    
    left_boxes = detect_and_save('test_data/left/000000.png', model, 'left_image.jpg', {0: 'person', 2: 'car'})

    P0 = get_projection_matrix('test_data/calib/000000.txt', 'P0')
    K, _, _, _, _, _, _ = cv2.decomposeProjectionMatrix(P0)

    distanceByHeight = DistanceBySize(K[0][0], K[1][1])

    for box in left_boxes:
        print(f"Class: {box.cls}, Confidence: {box.conf}, Box: {box.xyxy}")

        d = distanceByHeight.calculate(box)
        print(d)
