from typing import Optional
import timm
import torch
import torch.nn as nn

def build_model(model_name: str, num_classes: int, dropout: float = 0.2, pretrained: bool = True) -> nn.Module:
    """Build a timm model and replace classification head."""
    m = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=dropout)
    return m

def predict_proba(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        return torch.softmax(logits, dim=1)
