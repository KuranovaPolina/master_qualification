import torch

from PIL import Image

import matplotlib.pyplot as plt

class DistanceByZoeDepth:   
    def __init__(self, model_type = "ZoeD_NK"):
        self.model = torch.hub.load("isl-org/ZoeDepth", model_type, pretrained=True)
        self.model = self.model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()

    def calculate_depth_map(self, path):
        image_rgb = Image.open(path).convert("RGB")

        depth = self.model.infer_pil(image_rgb)

        return depth

def distance_by_zoe_depth(zoe_depth_model, boxes, left_img_path):
    depth_map = zoe_depth_model.calculate_depth_map(left_img_path)

    # plt.figure(figsize=(10, 5))
    # plt.imshow(depth_map, cmap='flag')
    # plt.title("Zoe depth map")
    # plt.show()

    distances = []

    for box in boxes:
        print(f"Class: {box.cls}, Confidence: {box.conf}, Box: {box.xywh}")

        # x = box.xywh.round().int().tolist()[0][0]
        # y = box.xywh.round().int().tolist()[0][1]
        # d = depth_map[y][x]

        x1 = box.xyxy.round().int().tolist()[0][0]
        y1 = box.xyxy.round().int().tolist()[0][1]
        x2 = box.xyxy.round().int().tolist()[0][2]
        y2 = box.xyxy.round().int().tolist()[0][2]

        object_map = depth_map[y1:y2, x1:x2]
        positive_values = object_map[object_map > 0]
        d = np.min(positive_values) if positive_values.size > 0 else 0

        distances.append(d)

    return distances

