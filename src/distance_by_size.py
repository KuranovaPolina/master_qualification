import cv2

from utils import get_projection_matrix

class ObjectType:
    def __init__(self, index, base_size_type, base_size):
        self.index = index # 0 / 2
        self.base_size_type = base_size_type # 0 - w, 1 - h
        self.base_size = base_size # int
    
    def print(self):
        print(f"Class: {self.index}, Type: {self.base_size_type}, Size: {self.base_size}")

class DistanceBySize:    
    def __init__(self, config_path, object_types = {0: ObjectType(0, 1, 1.8), 2: ObjectType(2, 1, 1.4)}):
        P0 = get_projection_matrix(config_path, 'P0')
        K, _, _, _, _, _, _ = cv2.decomposeProjectionMatrix(P0)
        
        self.fx = K[0][0]
        self.fy = K[1][1]
        self.object_types = object_types
    
    def calculate(self, box):
        object_type = self.object_types[box.cls.item()]
        object_type.print()

        if object_type.base_size_type == 0:
            return self.fx * object_type.base_size / (box.xywh[0][2].item())
        elif object_type.base_size_type == 1:
            return self.fy * object_type.base_size / (box.xywh[0][3].item())
        else:
            return 0
