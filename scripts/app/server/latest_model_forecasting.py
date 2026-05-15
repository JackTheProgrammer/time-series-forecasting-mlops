from yfinance import download
from datetime import datetime
from torch import tensor, float32
from torch.utils.data import Dataset, DataLoader

import os
from pathlib import Path
import joblib # Needed to load your saved scaler

class GoldStockPriceDataset(Dataset):
    def __init__(self, scaled_prices, window_size):
        # We now accept the pre-scaled numpy array
        self.scaled_prices = scaled_prices
        self.window_size = window_size

    def __len__(self):
        return len(self.scaled_prices) - self.window_size

    def __getitem__(self, idx):
        # Extract the window
        window = self.scaled_prices[idx:idx + self.window_size]
        return tensor(window, dtype=float32)

# 1. Download the raw data
todays_gold_stock_price = download("GC=F", start='2020-01-01', end=datetime.now().strftime("%Y-%m-%d"))
raw_close_prices = todays_gold_stock_price[["Close"]].values

# 2. CRITICAL STEP: Load your training scaler and apply it
# Replace 'path/to/your/saved_scaler.pkl' with your actual file path
SCALED_PKL_DIR = Path('scaled_transform')
# SCALED_PKL_DIR.mkdir(exist_ok=True)

scaled_pkls_paths = []

for pkl_file in SCALED_PKL_DIR.iterdir():
    if pkl_file.is_file() and pkl_file.suffix == '.pkl':
        scaled_pkls_paths.append(str(pkl_file))

latest_pkl_file = max(scaled_pkls_paths, key=os.path.getmtime)

SAVED_SCALE_PATH = Path(latest_pkl_file)

scaler = joblib.load(SAVED_SCALE_PATH)
scaled_stock_prices = scaler.transform(raw_close_prices)

# 3. Create Dataset and DataLoader
gold_stock_dataset = GoldStockPriceDataset(scaled_stock_prices, window_size=30)

# 4. CRITICAL STEP: shuffle=False for Time Series charting
gold_stock_dataloader = DataLoader(gold_stock_dataset, batch_size=32, shuffle=False)