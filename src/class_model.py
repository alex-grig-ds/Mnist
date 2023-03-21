
import timm
import torch.nn as nn

from config import *

class CustomModel(nn.Module):
    def __init__(self, target_size: int):
        super().__init__()
        self.model = timm.create_model(BASE_MODEL, pretrained=True, in_chans=1, num_classes=target_size)

    def forward(self, x):
        x = self.model(x)
        return x