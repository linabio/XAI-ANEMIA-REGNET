import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch
from typing import Tuple, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """Initialize the ImageProcessor with a target size."""
        self.target_size = target_size
        self.preprocess = self._create_preprocess_transform()

    def _create_preprocess_transform(self) -> transforms.Compose:
        """Create a preprocessing transform for model input."""
        return transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def load_image(self, image_path: str) -> np.ndarray:
        """Load an image from a file path."""
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image notv found: {image_path}")

        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Unsupported image format: {image_path}")
            return self._process_cv2_image(img)
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return np.zeros((*self.target_size, 3), dtype=np.float32)

    def process_array(self, image_array: np.ndarray) -> np.ndarray:
        """Process an image array."""
        try:
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            return self._process_cv2_image(image_array)
        except Exception as e:
            logger.error(f"Error processing array: {e}")
            return np.zeros((*self.target_size, 3), dtype=np.float32)

    def _process_cv2_image(self, img: np.ndarray) -> np.ndarray:
        """Process a CV2 image to the target format."""
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size)
        return img.astype(np.float32) / 255.0

    def preprocess_for_model(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """Preprocess an image for model input."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray((image * 255).astype(np.uint8))
        return self.preprocess(image)