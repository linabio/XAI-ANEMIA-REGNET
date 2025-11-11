import cv2
import numpy as np


def msrcp_enhance(image):
    img = image.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255.0
    return img.astype(np.uint8)
