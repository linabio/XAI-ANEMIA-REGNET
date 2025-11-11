import cv2
import numpy as np


def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    if image is None:
        return None
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    filtered = cv2.bilateralFilter(img_bgr, d, sigma_color, sigma_space)
    return cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
