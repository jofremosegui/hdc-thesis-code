# memory_benchmark_BSC.py
import torch
import torch.nn as nn
import psutil
import os
import time
import numpy as np
import gc

from torchhd import models, embeddings, functional

class HdcGenericEncoder(nn.Module):
    def __init__(self, input_size, out_dimension, ngrams=7, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.input_size = input_size
        self.keys = embeddings.Random(input_size, out_dimension, dtype=dtype, device=device, vsa="BSC")
        self.motion_embed = embeddings.Level(3000, out_dimension, dtype=dtype, low=-3.0, high=3.0, device=device, vsa="BSC")
        self.hr_embed = embeddings.Level(200, out_dimension, dtype=dtype, low=50, high=200, device=device, vsa="BSC")
        self.ngrams = ngrams
        self.device = device

    def batch_generic(self, id, levels, ngram):
        batch_size = levels.shape[0]
        multiset_list = []
        for b in range(batch_size):
            level = levels[b]
            b_levels = [
                functional.ngrams(level[0][i : i + ngram], ngram)
                for i in range(1, id.shape[0] - ngram + 1)
            ]
            if b_levels:
                b_levels = torch.stack(b_levels)
                multiset_list.append(functional.multiset(functional.bind(id[:-ngram], b_levels)).unsqueeze(0))
            else:
                multiset_list.append(functional.multiset(functional.bind(id, level)))
        return torch.stack(multiset_list)

    def forward(self, channels):
        motion = channels[:, :, :self.input_size - 1]
        hr = channels[:, :, self.input_size - 1].unsqueeze(-1)
        enc_motion = self.motion_embed(motion)
        enc_hr = self.hr_embed(hr)
        enc = torch.cat([enc_motion, enc_hr], dim=2)
        hvs = self.batch_generic(self.keys.weight, enc, self.ngrams)
        return functional.hard_quantize(functional.multiset(hvs))

class HdcModel(nn.Module):
    def __init__(self, input_size=7, out_dimension=5000, ngrams=7, device="cpu"):
        super().__init__()
        self.encoder = HdcGenericEncoder(input_size, out_dimension, ngrams, device=device)
        self.centroid = models.Centroid(out_dimension, 2, device=device)

    def forward(self, x):
        hv = self.encoder(x)
        norm_centroids = self.centroid.weight.clone()
        norm_centroids = norm_centroids / norm_centroids.norm(dim=1, keepdim=True).clamp(min=1e-12)
        sim = functional.dot_similarity(hv, norm_centroids)
        probs = torch.softmax(sim, dim=1)
        return probs[:, 1]  # return probability of class 1

# Main benchmark
def main():
    INPUT_SHAPE = (1, 800, 7)
    MODEL_PATH = "results/HDC_balanced_UNDERSAMPLING_TWEAKED_BSC/20250507_182624/model_20250508_023412.pt"
    device = torch.device("cpu")

    model = HdcModel(device=device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
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
