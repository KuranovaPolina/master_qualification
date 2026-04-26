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

        x = box.xywh.round().int().tolist()[0][0]
        y = box.xywh.round().int().tolist()[0][1]
        d = depth_map[y][x]

        distances.append(d)

    return distances

