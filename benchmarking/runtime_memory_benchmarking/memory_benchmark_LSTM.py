# memory_benchmark_gru.py
import torch
import torch.nn as nn
import psutil
import os
import time
import numpy as np
import gc

class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        bidirectional=True,
        pooling="attention",
    ):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pooling = pooling

        # bidirectionalの場合、出力の次元数が2倍になる
        self.direction_factor = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # Attention層（pooling='attention'の場合に使用）
        if pooling == "attention":
            self.attention = nn.Sequential(
                nn.Linear(hidden_size * self.direction_factor, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
            )

        # 全結合層の入力サイズを調整
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.direction_factor, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def attention_pooling(self, lstm_out):
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended_out = torch.sum(attention_weights * lstm_out, dim=1)
        return attended_out

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        pooled_output = lstm_out[:, -1, :]

        return self.fc(pooled_output)


# Main benchmark
def main():
    INPUT_SHAPE = (1, 800, 7)
    MODEL_PATH = "results/LSTM_DATA_AUGMENTATION/20250601_124653/model_20250601_143308.pt"
    device = torch.device("cpu")

    model = LSTMClassifier(input_size=INPUT_SHAPE[2])
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
