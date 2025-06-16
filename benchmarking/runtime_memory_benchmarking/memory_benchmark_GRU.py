# memory_benchmark_gru.py
import torch
import torch.nn as nn
import psutil
import os
import time
import numpy as np
import gc

# Define GRU model inline
class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.5, bidirectional=False, pooling="attention"):
        super(GRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pooling = pooling
        self.direction_factor = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
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
            nn.Sigmoid()
        )

    def attention_pooling(self, gru_out):
        weights = self.attention(gru_out)
        weights = torch.softmax(weights, dim=1)
        return torch.sum(weights * gru_out, dim=1)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        if self.pooling == "attention":
            pooled = self.attention_pooling(gru_out)
        else:
            pooled = gru_out[:, -1, :]
        return self.fc(pooled)

# Main benchmark
def main():
    INPUT_SHAPE = (1, 800, 7)
    MODEL_PATH = "results/GRU_DATA_AUGMENTATION/20250530_233726/model_20250531_195904.pt"
    device = torch.device("cpu")

    model = GRUClassifier(input_size=INPUT_SHAPE[2])
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
