from yfinance import download
from datetime import datetime
from torch import tensor, float32
from torch.utils.data import Dataset, DataLoader
# import numpy as np
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
try:
    scaler = joblib.load('path/to/your/saved_scaler.pkl')
    scaled_stock_prices = scaler.transform(raw_close_prices)
except FileNotFoundError:
    print("WARNING: Scaler not found. You MUST load the training scaler.")
    # Fallback ONLY for testing code execution (do not use in production)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_stock_prices = scaler.fit_transform(raw_close_prices)
    SCALED_PKL_DIR = Path('scaled_transform')
    SCALED_PKL_DIR.mkdir(exist_ok=True)
    joblib.dump(scaler, SCALED_PKL_DIR / f'{datetime.now().strftime('%Y-%m-%d')}_scaled_gold_stock.pkl')

# 3. Create Dataset and DataLoader
gold_stock_dataset = GoldStockPriceDataset(scaled_stock_prices, window_size=30)

# 4. CRITICAL STEP: shuffle=False for Time Series charting
gold_stock_dataloader = DataLoader(gold_stock_dataset, batch_size=32, shuffle=False)