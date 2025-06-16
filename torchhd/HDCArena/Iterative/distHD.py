import torch
from tqdm import tqdm
import torchmetrics
from collections import deque
import math


def train_distHD(
    train_loader,
    device,
    encode,
    model,
    iterations,
    lr=1,
    dimensions=10000,
):
    with torch.no_grad():
        for iter in range(iterations):
            for idx, (samples, labels) in enumerate(
                tqdm(train_loader, desc="Training")
            ):
                samples = samples.to(device)
                labels = labels.to(device)
                samples_hv = encode(samples)
                model.add_dist(samples_hv, labels, lr=lr)
                model.eval_dist(
                    samples_hv, labels, device, alpha=alpha, beta=beta, theta=theta
                )
            model.regenerate_dist(int(r * dimensions), encode, device)
    return iterations


def test_distHD(test_loader, device, encode, model, accuracy):
    model.normalize()
    with torch.no_grad():
        for samples, labels in tqdm(test_loader, desc="Testing"):
            samples = samples.to(device)

            samples_hv = encode(samples)
            outputs = model(samples_hv, dot=True)
            accuracy.update(outputs.to(device), labels.to(device))
