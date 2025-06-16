# memory_benchmark_hdc_hybrid.py

import torch
import torch.nn as nn
import numpy as np
import psutil
import os
import gc
import joblib  # or use pickle if needed
from torchhd import embeddings, functional
import sys

# --- HDC Encoder Class ---
class HdcGenericEncoder(nn.Module):
    def __init__(self, input_size, out_dimension, ngrams=7, dtype=torch.float32, device="cpu", vsa="MAP"):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.ngrams = ngrams
        self.keys = embeddings.Random(input_size, out_dimension, dtype=dtype, device=device, vsa=vsa)
        self.motion_embed = embeddings.Level(3000, out_dimension, low=-3.0, high=3.0, dtype=dtype, device=device, vsa=vsa)
        self.hr_embed = embeddings.Level(200, out_dimension, low=50, high=200, dtype=dtype, device=device, vsa=vsa)

    def batch_generic(self, id, levels, ngram):
        multiset_list = []
        for b in range(levels.shape[0]):
            level = levels[b]
            b_levels = [functional.ngrams(level[0][i:i+ngram], ngram)
                        for i in range(1, id.shape[0] - ngram + 1)]
            if b_levels:
                b_levels = torch.stack(b_levels)
                multiset_list.append(functional.multiset(functional.bind(id[:-ngram], b_levels)).unsqueeze(0))
            else:
                multiset_list.append(functional.multiset(functional.bind(id, level)))
        return torch.stack(multiset_list)

    def forward(self, x):
        motion = x[:, :, :-1]
        hr = x[:, :, -1].unsqueeze(-1)
        enc_motion = self.motion_embed(motion)
        enc_hr = self.hr_embed(hr)
        enc = torch.cat([enc_motion, enc_hr], dim=2)
        hvs = self.batch_generic(self.keys.weight, enc, self.ngrams)
        return functional.hard_quantize(functional.multiset(hvs))

# --- Main Benchmark Script ---
def main():
    INPUT_SHAPE = (1, 800, 7)
    ENCODER_PATH = "results/HDC_HYBRID_RANDOM_FOREST/20250508_225706/encoder.pt"  # Update this
    CLASSIFIER_PATH = "results/HDC_HYBRID_RANDOM_FOREST/20250508_225706/RandomForest_model.pkl"  # Update this
    device = torch.device("cpu")

    # Load encoder
    encoder = HdcGenericEncoder(input_size=7, out_dimension=5000, ngrams=7, device=device)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
    encoder.to(device)
    encoder.eval()

    # Load classifier
    classifier = joblib.load(CLASSIFIER_PATH)  # Or use pickle if needed

    # Prepare input
    sample_input = torch.randn(INPUT_SHAPE).to(device)
    with torch.no_grad():
        hv = encoder(sample_input).cpu().numpy()

    process = psutil.Process(os.getpid())
    gc.collect()
    mem_before = process.memory_info().rss

    # Perform classification
    _ = classifier.predict(hv)

    gc.collect()
    mem_after = process.memory_info().rss

    delta_mb = (mem_after - mem_before) / (1024 ** 2)
    total_mb = mem_after / (1024 ** 2)

    print(f"Δ Memory Used During Inference: {delta_mb:.2f} MB")
    print(f"Total Memory Usage After Inference: {total_mb:.2f} MB")

if __name__ == "__main__":
    main()
