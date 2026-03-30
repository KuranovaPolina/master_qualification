import sys
import torch
import cv2

from matplotlib import pyplot as plt  

from PIL import Image

class DistanceByZoeDepth:   
    def __init__(self, model_type = "ZoeD_NK"):
        self.model = torch.hub.load("isl-org/ZoeDepth", model_type, pretrained=True)
        self.model = self.model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()

    def calculate_depth_map(self, path):
        image_rgb = Image.open(path).convert("RGB")

        depth = self.model.infer_pil(image_rgb)

        return depth