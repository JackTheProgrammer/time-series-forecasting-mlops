import torch 
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.adam import Adam

import tqdm
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any
import shutil, copy, sys

sys.path.append('.')  # Add the parent directory to the system path
sys.path.append('./') # Add the current directory to the system path 

from src.evaluate import mse_loss
from src.architectures.lstm import GoldStockPriceLSTM
from src.architectures.gru import GoldStockPriceGRU
from src.architectures.conv1d import GoldStockPriceConv1D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocessed_data_path = f"data/processed/gold_data_preprocessed_{datetime.now().strftime('%Y-%m-%d')}.csv"
goldstock_preprocessed = read_csv(preprocessed_data_path, index_col='Date', parse_dates=True)

# [[[1], [2], [3], [4], [5]]] [6]
# [[[2], [3], [4], [5], [6]]] [7]
# [[[3], [4], [5], [6], [7]]] [8]

def create_sequences(scaled_data: np.ndarray, look_back=5):
    X = []
    y = []

    for i in range(len(scaled_data)-look_back):
        row = scaled_data[i:i+look_back] 
        X.append(row)
        label = scaled_data[i+look_back]
        y.append(label)

    return torch.tensor(np.array(X)).float(), torch.tensor(np.array(y)).float()

scaler = MinMaxScaler()
goldstock_preprocessed_scaled = scaler.fit_transform(goldstock_preprocessed)
X, y = create_sequences(goldstock_preprocessed_scaled, look_back=30)

train_size = int(0.8 * len(X))
val_size = int(0.1 * len(X))
test_size = len(X) - train_size - val_size

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

def validate_model(model: nn.Module, val_loader: DataLoader) -> float | Any:
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = mse_loss(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)

    avg_val_loss = val_loss / len(val_loader.dataset)
    return avg_val_loss

WORK_DIR = Path("model")
EPOCHS = 400

class EarlyStopper:
    def __init__(
        self, 
        mode='min', 
        architecture_name = ''
    ):
        self.save_path = f"{architecture_name}_goldstock_prices_{datetime.now().month}_{datetime.today().year}_best_model.pt"
        self.patience = int(EPOCHS * 0.1)  # Set patience to 10% of total epochs
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.best_model_wts = None
        self.early_stop = False

        self.checkpoint_path = WORK_DIR
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        if self.mode == 'min':
            self._score_compare_op = lambda score, best_score: score < best_score
        elif self.mode == 'max':
            self._score_compare_op = lambda score, best_score: score > best_score
        else:
            raise ValueError("Mode must be 'min' or 'max'")

    def __call__(self, current_metric_value, model):
        score = current_metric_value

        if self.best_score is None:
            self.best_score = score
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self._save_checkpoint()
        elif self._score_compare_op(score, self.best_score):
            self.best_score = score
            self.counter = 0
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self._save_checkpoint()
        else:
            self.counter += 1
            print(f"Validation metric did not improve. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print("Patience reached!! Early stopping triggered.")
        
        return self.early_stop
    
    def _save_checkpoint(self):
        """Saves the current model state dictionary to the defined path."""
        checkpoint_dir = self.checkpoint_path/f"{self.save_path}"
        torch.save(self.best_model_wts, checkpoint_dir)
        print(f"New best model saved to {checkpoint_dir} (Score: {self.best_score:.4f})")

    def load_best_weights(self, model: nn.Module):
        """Loads the best saved weights from disk into the model instance."""
        checkpoint_dir = self.checkpoint_path/f"{self.save_path}" 
        if checkpoint_dir.exists():
            model.load_state_dict(torch.load(checkpoint_dir, map_location=device))
            print(f"Loaded best weights from disk: {self.save_path}")
        else:
            print(f"Checkpoint file not found at {self.save_path}.")
    
    def get_saved_model_path(self):
        """Returns the full path to the saved model checkpoint."""
        return self.checkpoint_path/f"{self.save_path}"

def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer) -> float | Any:
    model.train()
    train_loss = 0.0
    
    for X_batch, y_batch in tqdm.tqdm(train_loader, desc="Training"):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = mse_loss(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * X_batch.size(0)

    avg_train_loss = train_loss / len(train_loader.dataset)
    return avg_train_loss

def train_model(
    model: nn.Module,
    architecture_name: str,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    batch_size: int = 32,
    learning_rate: float = 0.001
) -> tuple[nn.Module, object]:
    model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Create DataLoader for training and validation sets
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the early stopper
    early_stopper = EarlyStopper(mode='min', architecture_name=architecture_name)

    for epoch in range(EPOCHS):
        avg_train_loss = train_epoch(model, train_loader, optimizer)
        avg_val_loss = validate_model(model, val_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Check for early stopping
        if early_stopper(avg_val_loss, model):
            break
    
    early_stopper.load_best_weights(model)
    return model, early_stopper.get_saved_model_path()

models_with_names = [
    (GoldStockPriceLSTM(), "LSTM"),
    (GoldStockPriceGRU(), "GRU"),
    (GoldStockPriceConv1D(), "Conv1D"),
]

trained_models = dict[str, tuple[nn.Module, object]]()

for train_model_instance, architecture_name in models_with_names:
    print(f"\nStarting training for {architecture_name} model...")
    trained_model, saved_wts_path = train_model(
        model=train_model_instance,
        architecture_name=architecture_name,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=32,
        learning_rate=0.001
    )
    trained_models[architecture_name] = (trained_model, saved_wts_path)
    
def test_serialized_model(model: nn.Module, test_loader: DataLoader) -> float | Any:
    model.eval()
    
    test_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = mse_loss(outputs, y_batch)
            test_loss += loss.item() * X_batch.size(0)
    
    avg_test_loss = test_loss / len(test_loader.dataset)
    return avg_test_loss

# testing every trained model on the test set and printing the results
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

best_test_loss = float('inf')
best_model_details = None

for model_name, (model_instance, saved_wts_path) in trained_models.items():
    print(f"\nTesting {model_name} model...")
    test_loss = test_serialized_model(model_instance, test_loader)
    print(f"{model_name} Test Loss: {test_loss:.4f}")
    
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_model_details = (model_name, saved_wts_path)

print(f"\nWinner Model: {best_model_details[0]} with Test Loss: {best_test_loss:.4f} at the path: {best_model_details[1]}")

WINNER_DIR = Path(f"../winner_models/{datetime.today().year}")
WINNER_DIR.mkdir(parents=True, exist_ok=True)

# Copy the best model's checkpoint to the winner directory
best_model_checkpoint_path = Path(best_model_details[1])
destination_path = WINNER_DIR / best_model_checkpoint_path.name
best_model_checkpoint_path.rename(destination_path)
print(f"Best model checkpoint moved to: {destination_path}")

# truncating the entire `WORK_DIR`

for item in WORK_DIR.iterdir():
    if item.is_file():
        item.unlink()
    elif item.is_dir():
        shutil.rmtree(item)

print(f"Truncated the {WORK_DIR} to free up space and maintain cleanliness.")