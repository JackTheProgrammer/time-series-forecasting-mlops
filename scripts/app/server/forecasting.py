from yfinance import download
from datetime import datetime
from torch import tensor, device, float32, cuda
from torch.utils.data import Dataset, DataLoader

device = device("cuda" if cuda.is_available() else "cpu")

def preprocess_stock_price_data():
    todays_gold_stock_price = download("GC=F", start=datetime(2010, 1, 1), end=datetime.now())
    stock_price_data = todays_gold_stock_price[["Close"]]
    stock_price_data.drop([0,1], inplace=True)
    stock_price_data.rename(columns={'Price': 'Date', 'Close': 'closing_price'}, inplace=True)
    stock_price_data.set_index('Date', inplace=True)
    stock_price_data.index = stock_price_data.index.astype('datetime64[ns]')
    return stock_price_data

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
        return tensor(stock_price_values[idx:idx + self.window_size], dtype=float32).to(device)

todays_gold_stock_price = preprocess_stock_price_data()
gold_stock_dataset = GoldStockPriceDataset(todays_gold_stock_price, window_size=30)
gold_stock_dataloader = DataLoader(gold_stock_dataset, batch_size=32, shuffle=True)

# testing the dataloader
for batch in gold_stock_dataloader:
    print(batch)
    break