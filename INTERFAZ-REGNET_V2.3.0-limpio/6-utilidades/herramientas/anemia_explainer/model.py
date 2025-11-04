import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from torchvision.models import RegNet_Y_400MF_Weights, RegNet_Y_800MF_Weights
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    backbone_name: str = 'regnet_y_400mf'
    input_size: Tuple[int, int] = (128, 128)
    num_classes: int = 2
    dropout_rates: List[float] = None
    hidden_dims: List[int] = None

    def __post_init__(self):
        self.dropout_rates = self.dropout_rates or [0.6, 0.3]
        self.hidden_dims = self.hidden_dims or [128]

class BaseModel(ABC, torch.nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        pass

class RegNetBinaryClassifier(BaseModel):
    def __init__(self,
                 config: Optional[ModelConfig] = None,
                 weights_path: Optional[str] = None,
                 device: str = 'cpu'):
        """Initialize the RegNet model with customizable configuration."""
        super().__init__()
        self.config = config or ModelConfig()
        self.device = device
        self.num_classes = self.config.num_classes

        if self.config.backbone_name == 'regnet_y_400mf':
            self.backbone = models.regnet_y_400mf(weights=RegNet_Y_400MF_Weights.DEFAULT)
            self.feature_dim = 440
        elif self.config.backbone_name == 'regnet_y_800mf':
            self.backbone = models.regnet_y_800mf(weights=RegNet_Y_800MF_Weights.DEFAULT)
            self.feature_dim = 784
        else:
            raise ValueError(f"Unsupported backbone: {self.config.backbone_name}")

        self.pool = nn.AdaptiveAvgPool2d(1)

        layers = []
        in_dim = self.feature_dim
        for hidden_dim, dropout_rate in zip(self.config.hidden_dims, self.config.dropout_rates):
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.num_classes))
        self.classifier = nn.Sequential(*layers)

        if weights_path:
            self._load_weights(weights_path)

        self.to(device)
        logger.info(f"Model loaded on {device}")

    def _load_weights(self, weights_path: str) -> None:
        """Load model weights from a file."""
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys}")
            logger.info(f"Weights loaded from: {weights_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading weights: {e}") from e

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        with torch.inference_mode():
            x = self.backbone.stem(x)
            x = self.backbone.trunk_output(x)
            x = self.pool(x).flatten(1)
            x = self.classifier(x)
            return x  # No squeeze for multiclass

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities."""
        with torch.inference_mode():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            del logits
            if 'cuda' in self.device:
                torch.cuda.empty_cache()
            return probs