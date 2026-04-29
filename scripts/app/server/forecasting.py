from yfinance import download
from datetime import datetime
from torch import tensor, float32
from torch.utils.data import Dataset, DataLoader

# I intend to make a dataset for forecasting purpose, this 
# means that there're no X, y based sequences creation
class GoldStockPriceDataset(Dataset):
    def __init__(self, stock_prices, window_size):
        self.stock_prices = stock_prices
        self.window_size = window_size

    def __len__(self):
        return len(self.stock_prices) - self.window_size

    def __getitem__(self, idx):
        stock_price_values = self.stock_prices['Close'].values
        return tensor(stock_price_values[idx:idx + self.window_size], dtype=float32)

todays_gold_stock_price = download("GC=F", start='2020-01-01', end=datetime.now().strftime("%Y-%m-%d"))
todays_gold_stock_price = todays_gold_stock_price[["Close"]]

gold_stock_dataset = GoldStockPriceDataset(todays_gold_stock_price, window_size=30)
gold_stock_dataloader = DataLoader(gold_stock_dataset, batch_size=32, shuffle=True)