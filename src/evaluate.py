from torch import nn
from typing import Any

def mse_loss(outputs, y_batch) -> Any:
    criterion = nn.MSELoss()
    return criterion(outputs.squeeze(), y_batch.squeeze())