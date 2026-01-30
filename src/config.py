from dataclasses import dataclass

@dataclass(frozen=True)
class TrainConfig:
    img_size: int = 224
    batch_size: int = 32
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    dropout: float = 0.2
    seed: int = 42
