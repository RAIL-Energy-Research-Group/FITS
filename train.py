import torch
import random
import numpy as np
import pandas as pd
from dataset import TimeSeriesDataset, create_windows
from torch.utils.data import DataLoader
from model import FITS
from early_stopping import EarlyStopping
from torch import nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")


SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


DATA_PATH = "transformer/transformer_82T1.csv"

data = pd.read_csv(DATA_PATH)
data['DATE'] = pd.to_datetime(data['DATE'])
#  This helps us enure the dataset is sorted by date
data.set_index('DATE', inplace=True)

feature_columns = ['82T1_BANK (A)', '82T1_BANK (MW)', '82T1_BANK (MVA)', '82T1_TEMPERATURE_WDG_1 (VALUE)', '82T1_TEMPERATURE_OIL_1 (VALUE)']
series = data[feature_columns].values
num_channels = series.shape[1]


# Channel to plot for visualization
channel_to_plot_name = '82T1_TEMPERATURE_WDG_1 (VALUE)'
channel_to_plot_idx = feature_columns.index(channel_to_plot_name)

# Hyperparameters
LOOK_BACK = 24 * 3  # 7 days
HORIZON = 24  # 1 day
COF = 20  # Cutoff frequency this must be < look_back // 2 + 1
"""
  This is BCos converting from time domain
  to frequency domain results 
  in a frequency domain with length look_back // 2 + 1
"""
BATCH_SIZE = 32
EPOCHS = 500
LEARNING_RATE = 0.01

X, y = create_windows(series, LOOK_BACK, HORIZON) # Retuns numpy

# Convert to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

train_dataset = TimeSeriesDataset(X, y)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = FITS(LOOK_BACK, HORIZON, COF, num_channels).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
early_stopping = EarlyStopping(patience=5, verbose=True, path='transformer_82T1_WDG.pt')

for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            
            batch_x_permuted = batch_x.permute(0, 2, 1)

            
            output_permuted = model(batch_x_permuted)
            
            output = output_permuted.permute(0, 2, 1)
            
            backcast = output[:, :LOOK_BACK, :]
            forecast = output[:, -HORIZON:, :]
            # print(forecast)
            loss_backcast = criterion(backcast, batch_x)
            loss_forecast = criterion(forecast, batch_y)
            loss = loss_backcast + loss_forecast
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}')
        
        early_stopping(avg_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
        
# Load the best model
model.load_state_dict(torch.load('transformer_82T1_WDG.pt'))
print("\nLoaded best model from checkpoint for evaluation.")

model.eval()
with torch.no_grad():
    X_permuted = X.to(device).permute(0, 2, 1)
    outputs_permuted = model(X_permuted)
    outputs = outputs_permuted.permute(0, 2, 1)
    
    forecasts = outputs[:, -HORIZON:, :]
    test_loss = criterion(forecasts, y.to(device))
    print(f'Final Test Loss (MSE on all channels): {test_loss.item():.4f}')

# Test the model with 6 random samples and plot the results
sample_indices = random.sample(range(len(X)), 6)
print("Sample indices for testing:", sample_indices)

plt.figure(figsize=(18, 12))
for i, sample_index in enumerate(sample_indices):
    sample_x = X[sample_index].unsqueeze(0).to(device)  # Add batch dimension
    sample_y = y[sample_index]

    sample_x_permuted = sample_x.permute(0, 2, 1)
    sample_output_permuted = model(sample_x_permuted)
    sample_output = sample_output_permuted.permute(0, 2, 1).detach().squeeze(0).cpu().numpy()

    forecast_plot = sample_output[-HORIZON:, channel_to_plot_idx]
    backcast_plot = sample_output[:LOOK_BACK, channel_to_plot_idx]
    
    input_plot = sample_x.squeeze(0).cpu().numpy()[:, channel_to_plot_idx]
    actual_plot = sample_y.numpy()[:, channel_to_plot_idx]

    plt.subplot(3, 2, i + 1)
    plt.plot(range(LOOK_BACK), input_plot, label='Input', color='blue')
    plt.plot(range(LOOK_BACK), backcast_plot, label='Backcast', color='red', linestyle='--')
    plt.plot(range(LOOK_BACK, LOOK_BACK + HORIZON), forecast_plot, label='Forecast', color='orange')
    plt.plot(range(LOOK_BACK, LOOK_BACK + HORIZON), actual_plot, label='Actual', color='green')
    
    plt.title(f'Sample {sample_index} - Channel: {channel_to_plot_name}')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
