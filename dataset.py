from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
        
def create_windows(series, look_back, horizon):
    X, y = [], []
    for i in range(len(series) - look_back - horizon + 1):
        X.append(series[i:i + look_back])
        y.append(series[i + look_back:i + look_back + horizon])
    return np.array(X), np.array(y)

