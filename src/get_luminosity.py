import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_illumination(img):
    b, g, r = cv2.split(img)
    # Y = 0.114*B + 0.587*G + 0.299*R
    luminance = 0.114 * b.astype(np.float32) + 0.587 * g.astype(np.float32) + 0.299 * r.astype(np.float32)
    luminance = np.clip(luminance, 0, 255).astype(np.uint8)
        
    flat_lum = luminance.ravel()

    median_lum = float(np.median(flat_lum))

    return median_lum

def get_luminosity(img_path, boxes):
    img_bgr = cv2.imread(img_path)

    luminosities = []

    for box in boxes:
        print(f"Class: {box.cls}, Confidence: {box.conf}, Box: {box.xyxy}")

        x1 = box.xyxy.round().int().tolist()[0][0]
        y1 = box.xyxy.round().int().tolist()[0][1]
        x2 = box.xyxy.round().int().tolist()[0][2]
        y2 = box.xyxy.round().int().tolist()[0][3]

        luminosities.append(analyze_illumination(img_bgr[y1:y2, x1:x2]))

    return luminosities
