import torch
from torch import nn
from datetime import datetime
import os, sys

sys.path.append('.') # to ensure we can import from the src directory regardless of where this script is run from
sys.path.append('./') # to register the current directory for imports as well, just in case

torch.manual_seed(42)

winner_root_dir = f'winner_models\\{datetime.today().year}'

saved_wts_paths = []

for model_dir in os.listdir(winner_root_dir):
    model_path = os.path.join(winner_root_dir, model_dir)
    if os.path.isfile(model_path) and model_path.endswith('.pt'):
        print(f"Loaded model from: {model_path}")
        saved_wts_paths.append(model_path)
    else:
        continue

# to get the latest of the winners, i'll pick the one with the most recent modified time
latest_model_path = max(saved_wts_paths, key=os.path.getmtime)
print(f"Latest model path: {latest_model_path}")

# getting the architecture name from the latest model path
latest_model_name = os.path.basename(latest_model_path)
architecture_name = latest_model_name.split('_goldstock_prices')[0]
print("Architecture name: ", architecture_name)

loaded_model = None
if architecture_name == 'LSTM':
    from scripts.architectures.lstm import GoldStockPriceLSTM
    loaded_model = GoldStockPriceLSTM()
if architecture_name == 'GRU':
    from scripts.architectures.gru import GoldStockPriceGRU
    loaded_model = GoldStockPriceGRU()
if architecture_name == 'Conv1D':
    from scripts.architectures.conv1d import GoldStockPriceConv1D
    loaded_model = GoldStockPriceConv1D()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loaded_model.to(device)
loaded_model.load_state_dict(torch.load(latest_model_path, map_location=device))

def forecast_next_price(input_sequence: torch.Tensor):
    loaded_model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        predicted_price = loaded_model(input_sequence)  # Get the predicted price
    return predicted_price  # Return the predicted price as a scalar

# dummy stock data for testing
dummy_data = torch.rand(32, 30, 1)  # Shape: (batch_size, sequence_length, num_features)
dummy_data = dummy_data.to(device)
print("Dummy data shape: ", dummy_data.shape)
print("Original dummy data: ", dummy_data)
print("Dummy data as flattened: ", dummy_data.flatten())

forecasted_price = forecast_next_price(dummy_data)
print("Forecasted price shape: ", forecasted_price.shape if isinstance(forecasted_price, torch.Tensor) else "Not a tensor")
print("Forecasted price: ", forecasted_price)
print("Forecasted next price flattened: ", forecasted_price.flatten())