import torch
from torch import nn

class GoldStockPriceGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(GoldStockPriceGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        device = x.device  # Get the device of the input tensor
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        
        # Forward propagate the GRU
        out, _ = self.gru(x, h0)
        
        # We only care about the last time step's output for the prediction
        out = self.fc(out[:, -1, :])
        
        # out shape will be (batch_size, 1) which is the predicted price
        # for the next day
        return out