# memory_benchmark.py
import torch
import torch.nn as nn
import psutil
import os
import time
import numpy as np
import gc

# Define model inline
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1, dropout=0.3, bidirectional=True, pooling="attention"):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pooling = pooling
        self.direction_factor = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        if pooling == "attention":
            self.attention = nn.Sequential(
                nn.Linear(hidden_size * self.direction_factor, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
            )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.direction_factor, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def attention_pooling(self, lstm_out):
        weights = self.attention(lstm_out)
        weights = torch.softmax(weights, dim=1)
        return torch.sum(weights * lstm_out, dim=1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        if self.pooling == "attention":
            pooled = self.attention_pooling(lstm_out)
        else:
            pooled = lstm_out[:, -1, :]
        return self.fc(pooled)

# Main benchmark
def main():
    INPUT_SHAPE = (1, 800, 7)  # adjust to your case
    MODEL_PATH = "results/biLSTM_DATA_AUGMENTATION/20250520_183833/model_20250521_024421.pt"
    device = torch.device("cpu")

    model = LSTMClassifier(input_size=INPUT_SHAPE[2], pooling="attention")
    state_dict = torch.load(MODEL_PATH, map_location=device)
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()

    sample_input = torch.randn(INPUT_SHAPE).to(device)
    process = psutil.Process(os.getpid())

    gc.collect()
    mem_before = process.memory_info().rss

    with torch.no_grad():
        _ = model(sample_input)

    gc.collect()
    mem_after = process.memory_info().rss

    delta_mb = (mem_after - mem_before) / (1024 ** 2)
    total_mb = mem_after / (1024 ** 2)

    print(f"Δ Memory Used During Inference: {delta_mb:.2f} MB")
    print(f"Total Memory Usage After Inference: {total_mb:.2f} MB")

if __name__ == "__main__":
    main()
