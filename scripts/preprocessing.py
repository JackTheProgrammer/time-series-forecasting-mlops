from pandas import read_csv, to_datetime
from yfinance import download
from datetime import datetime

data = download("GC=F", start='2010-01-01', end=datetime.now().strftime("%Y-%m-%d"))
data = data[["Close"]]

raw_data_path = f"data/raw/gold_price_data_{datetime.today().strftime('%Y-%m-%d')}.csv"
data.to_csv(raw_data_path, index=True)

goldstock_raw_data = read_csv(raw_data_path)
print("Column names:\n", goldstock_raw_data.columns)
goldstock_raw_data.drop([0,1], inplace=True)
goldstock_raw_data.rename(columns={'Price': 'Date', 'Close': 'closing_price'}, inplace=True)
print("Column renamed names:\n", goldstock_raw_data.columns)
goldstock_raw_data.set_index('Date', inplace=True)
print("Index is: ", goldstock_raw_data.index)

# goldstock_raw_data.set_index('Date', inplace=True)
goldstock_raw_data.index = to_datetime(goldstock_raw_data.index)

type_mapping = {
    'closing_price': 'float64'
}
goldstock_raw_data = goldstock_raw_data.astype(type_mapping)

preprocessed_data_path = f"data/processed/gold_data_preprocessed_{datetime.now().strftime('%Y-%m-%d')}.csv"
goldstock_raw_data.to_csv(preprocessed_data_path, index_label='Date')