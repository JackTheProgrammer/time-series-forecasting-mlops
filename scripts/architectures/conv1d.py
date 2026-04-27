from torch import nn
import torch.nn.functional as F

class GoldStockPriceConv1D(nn.Module):
    def __init__(self, input_size=1, num_filters=64, kernel_size=3):
        super(GoldStockPriceConv1D, self).__init__()
        self.input_size = input_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv1d = nn.Conv1d(self.input_size, self.num_filters, self.kernel_size)
        self.fc = nn.Linear(self.num_filters, 1)

    def forward(self, x):
        # x shape is (batch_size, seq_len, input_size)
        # We need to permute it to (batch_size, input_size, seq_len) for Conv1D
        x = x.permute(0, 2, 1)
        
        # to acquire the hidden details from the time dimension,
        # we apply a 1D convolution followed by a ReLU activation
        out = F.relu(self.conv1d(x))
        
        # Global average pooling over the time dimension
        # It's done in order to reduce the output of the
        # convolutional layer to a fixed size by taking the average 
        # across the time dimension, which allows us to feed it into 
        # the fully connected layer. The unsqueeze(2) is used to remove
        # the extra dimension after pooling; which is 'seq_len' in this case.
        out = F.adaptive_avg_pool1d(out, 1).squeeze(2)
        
        # Pretty much the same as before, we take the output of the last time step
        # like the tensorflow's Dense layer with 'last' activation, 
        # and feed it into a fully connected layer
        out = self.fc(out)
        
        # out shape will be (batch_size, 1) which is the predicted price
        # for the next day
        return out