import sys
import torch
import cv2

from matplotlib import pyplot as plt  

from PIL import Image

class DistanceByZoeDepth:   
    def __init__(self, model_type = "ZoeD_NK"):
        self.model = torch.hub.load("isl-org/ZoeDepth", model_type, pretrained=True)
        self.model = self.model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()

    def show_image(self, img):
        plt.figure(figsize=(10, 5))
        plt.imshow(img)
        plt.show()

    def calculate_depth_map(self, path):
        image_rgb = Image.open(path).convert("RGB")

        self.show_image(image_rgb)

        depth = self.model.infer_pil(image_rgb)

        self.show_image(depth)

        return depth