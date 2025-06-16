# memory_benchmark.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import psutil
import os
import time
import numpy as np
import gc

# Define CNN model inline
class CNNClassifier(nn.Module):
    def __init__(self, input_size, num_filters=32, kernel_sizes=[3, 5, 7], dropout=0.1, pooling="avg"):
        super(CNNClassifier, self).__init__()
        self.pooling = pooling
        
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=k, padding='same')
            for k in kernel_sizes
        ])
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters) for _ in kernel_sizes
        ])
        
        self.fc = nn.Sequential(
            nn.Linear(num_filters * len(kernel_sizes), 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.transpose(1, 2)  # [B, T, F] -> [B, F, T]
        conv_outputs = []
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            out = F.relu(bn(conv(x)))
            if self.pooling == "avg":
                pooled = F.adaptive_avg_pool1d(out, 1).squeeze(2)
            elif self.pooling == "max":
                pooled = F.adaptive_max_pool1d(out, 1).squeeze(2)
            else:
                pooled = F.adaptive_max_pool1d(out, 1).squeeze(2)  # fallback
            conv_outputs.append(pooled)
        combined = torch.cat(conv_outputs, dim=1)
        return self.fc(combined)

# Main benchmark
def main():
    INPUT_SHAPE = (1, 800, 7)
    MODEL_PATH = "results/1D-CNN_DATA_AUGMENTATION/20250528_193914/model_20250528_201856.pt"
    device = torch.device("cpu")

    model = CNNClassifier(input_size=INPUT_SHAPE[2])
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
