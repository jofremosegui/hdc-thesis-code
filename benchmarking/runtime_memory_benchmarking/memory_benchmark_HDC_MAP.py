# memory_benchmark_hrr.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import psutil
import os
import numpy as np
import gc

from torchhd import models, embeddings, functional

# --- HDC Model for HRR ---
class HdcGenericEncoder(nn.Module):
    def __init__(self, input_size, out_dimension, ngrams=7, dtype=torch.float32, device="cpu", vsa="MAP"):
        super().__init__()
        self.device = device
        self.vsa = vsa
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
        return functional.multiset(hvs)

class HdcModel(nn.Module):
    def __init__(self, input_size, out_dimension, ngrams=7, dtype=torch.float32, device="cpu", vsa="MAP"):
        super().__init__()
        self.encoder = HdcGenericEncoder(input_size, out_dimension, ngrams, dtype, device, vsa)
        self.centroid = models.Centroid(out_dimension, 2, dtype=dtype, device=device)

    def vector_norm(self, x, p=2, dim=None, keepdim=False):
        return torch.pow(torch.sum(torch.abs(x) ** p, dim=dim, keepdim=keepdim), 1 / p)

    def normalized_inference(self, input, dot=False):
        weight = self.centroid.weight.detach().clone()
        norms = self.vector_norm(weight, p=2, dim=1, keepdim=True)
        norms.clamp_(min=1e-12)
        weight.div_(norms)
        return functional.dot_similarity(input, weight) if dot else functional.cosine_similarity(input, weight)

    def forward(self, x):
        hv = self.encoder(x)
        sim = self.normalized_inference(hv, dot=True)
        return F.softmax(sim, dim=1)[:, 1]

# --- Main Benchmark ---
def main():
    INPUT_SHAPE = (1, 800, 7)
    MODEL_PATH = "results/HDC_balanced_UNDERSAMPLING_MAP/20250502_234513/model_20250503_041824.pt"  # Update this
    device = torch.device("cpu")

    model = HdcModel(input_size=INPUT_SHAPE[2], out_dimension=5000, ngrams=7, device=device, vsa="MAP")
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
