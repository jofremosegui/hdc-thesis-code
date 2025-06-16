import copy

import torch
import torch.nn as nn
import torch.utils.data as data
from torch import Tensor
from tqdm import tqdm
import torchmetrics
import torchhd
from torchhd.datasets import HDCArena, UCIClassificationBenchmark
from torchhd import embeddings
from torchhd.models import Centroid, CentroidMiss
import time
import csv
import torch.nn.functional as F


# Function for performing min-max normalization of the input data samples
def create_min_max_normalize(min: Tensor, max: Tensor):
    def normalize(input: Tensor) -> Tensor:
        return torch.nan_to_num((input.to(device) - min) / (max - min))

    return normalize


# Specify device to be used for Torch.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))
# Specifies batch size to be used for the model.


class Encoder(nn.Module):
    def __init__(self, size, dimensions, encoding):
        super(Encoder, self).__init__()
        self.encoding = encoding
        if self.encoding == "bundle":
            self.symbol = embeddings.Random(size, dimensions)
        if self.encoding == "hashmap":
            levels = 100
            self.keys = embeddings.Random(size, dimensions)
            self.values = embeddings.Level(levels, dimensions)
        if self.encoding == "ngram":
            self.symbol = embeddings.Random(size, dimensions)
        if self.encoding == "sequence":
            self.symbol = embeddings.Random(size, dimensions)
        if self.encoding == "random":
            self.embed = embeddings.Projection(size, dimensions)
        if self.encoding == "sinusoid":
            self.embed = embeddings.Projection(size, size * 2)
            self.embedd = embeddings.Sinusoid(size * 2, dimensions)
        if self.encoding == "density":
            self.embed = embeddings.Density(size, dimensions)
        if self.encoding == "flocet":
            self.embed = embeddings.Projection(size, size * 2)
            self.embed2 = embeddings.DensityFlocet(size * 2, dimensions)
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.flatten(x).float()
        if self.encoding == "bundle":
            sample_hv = torchhd.multiset(self.symbol(x.long()))
        if self.encoding == "hashmap":
            sample_hv = torchhd.hash_table(self.keys.weight, self.values(x))
        if self.encoding == "ngram":
            sample_hv = torchhd.ngrams(self.symbol(x.long()), n=3)
        if self.encoding == "sequence":
            sample_hv = torchhd.ngrams(self.symbol(x.long()), n=1)
        if self.encoding == "random":
            sample_hv = self.embed(x).sign()
        if self.encoding == "sinusoid":
            sample_hv = self.embedd(self.embed(x)).sign()
        if self.encoding == "density":
            sample_hv = self.embed(x).sign()
        if self.encoding == "flocet":
            sample_hv = self.embed2(self.embed(x)).sign()
        return torchhd.hard_quantize(sample_hv)

    def neural_regeneration(self, idx):
        nn.init.normal_(self.embed.weight[idx], 0, 1)
        self.embed.weight[idx] = F.normalize(self.embed.weight)[idx]


# Get an instance of the UCI benchmark
benchmark = UCIClassificationBenchmark(
    "/Users/verges/Documents/PhD/TorchHd/torchhd/examples/data", download=True
)
# Perform evaluation
results_file = (
    "/Users/verges/Documents/PhD/TorchHd/torchhd/examples/results/results"
    + str(time.time())
    + ".csv"
)

with open(results_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "Name",
            "Accuracy",
            "AccuracyMiss",
            "Time",
            "Dimensions",
            "Method",
            "Encoding",
            "Iterations",
            "Retrain",
        ]
    )


def exec_arena(
    method="add",
    encoding="density",
    iterations=1,
    retrain=False,
    dimensions=10,
    repeats=1,
    batch_size=1,
):
    iterations -= 1
    for dataset in benchmark.datasets():
        for r in range(repeats):
            print(dataset.name)
            if dataset.name == "EuropeanLanguages":
                from torchhd.datasets import EuropeanLanguages as Languages

                MAX_INPUT_SIZE = 128
                PADDING_IDX = 0

                ASCII_A = ord("a")
                ASCII_Z = ord("z")
                ASCII_SPACE = ord(" ")

                def char2int(char: str) -> int:
                    """Map a character to its integer identifier"""
                    ascii_index = ord(char)

                    if ascii_index == ASCII_SPACE:
                        # Remap the space character to come after "z"
                        return ASCII_Z - ASCII_A + 1

                    return ascii_index - ASCII_A

                def transform(x: str) -> torch.Tensor:
                    char_ids = x[:MAX_INPUT_SIZE]
                    char_ids = [char2int(char) + 1 for char in char_ids.lower()]

                    if len(char_ids) < MAX_INPUT_SIZE:
                        char_ids += [PADDING_IDX] * (MAX_INPUT_SIZE - len(char_ids))

                    return torch.tensor(char_ids, dtype=torch.long)

                num_feat = MAX_INPUT_SIZE

                train_ds = Languages(
                    "../data", train=True, transform=transform, download=True
                )
                train_loader = data.DataLoader(
                    train_ds, batch_size=BATCH_SIZE, shuffle=True
                )

                test_ds = Languages(
                    "../data", train=False, transform=transform, download=True
                )
                test_loader = data.DataLoader(
                    test_ds, batch_size=BATCH_SIZE, shuffle=False
                )
                num_classes = len(train_ds.classes)

            elif dataset.name in ["PAMAP", "EMGHandGestures"]:
                if dataset.name == "EMGHandGestures":
                    num_feat = dataset.train[0][0].size(-1) * dataset.train[0][0].size(
                        -2
                    )
                else:
                    num_feat = dataset.train[0][0].size(-1)

                # Number of classes in the dataset.
                num_classes = len(dataset.train.classes)
                # Number of training samples in the dataset.
                num_train_samples = len(dataset.train)

                # Get values for min-max normalization and add the transformation
                min_val = torch.min(dataset.train.data, 0).values.to(device)
                max_val = torch.max(dataset.train.data, 0).values.to(device)
                transform = create_min_max_normalize(min_val, max_val)
                dataset.train.transform = transform

                train_size = int(len(dataset.train) * 0.7)
                test_size = len(dataset.train) - train_size
                train_ds, test_ds = data.random_split(
                    dataset.train, [train_size, test_size]
                )

                train_loader = data.DataLoader(
                    train_ds, batch_size=batch_size, shuffle=True
                )
                test_loader = data.DataLoader(test_ds, batch_size=batch_size)
            else:
                # Number of features in the dataset.
                if dataset.name not in ["MNIST", "CIFAR10"]:
                    num_feat = dataset.train[0][0].size(-1)
                else:
                    if dataset.name == "MNIST":
                        num_feat = dataset.train[0][0].size(-1) * dataset.train[0][
                            0
                        ].size(-1)
                    elif dataset.name == "CIFAR10":
                        num_feat = 3072
                # Number of classes in the dataset.
                num_classes = len(dataset.train.classes)
                # Number of training samples in the dataset.
                num_train_samples = len(dataset.train)
                # Get values for min-max normalization and add the transformation

                if dataset.name not in ["MNIST", "CIFAR10"]:
                    min_val = torch.min(dataset.train.data, 0).values.to(device)
                    max_val = torch.max(dataset.train.data, 0).values.to(device)
                    transform = create_min_max_normalize(min_val, max_val)
                    dataset.train.transform = transform
                    dataset.test.transform = transform

                # Set up data loaders
                train_loader = data.DataLoader(
                    dataset.train, batch_size=batch_size, shuffle=True
                )
                test_loader = data.DataLoader(dataset.test, batch_size=batch_size)

            # Run for the requested number of simulations

            encode = Encoder(num_feat, dimensions, encoding)
            encode = encode.to(device)

            model = CentroidMiss(dimensions, num_classes, num_train_samples)
            model = model.to(device)

            t = time.time()

            accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
            accuracy2 = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
            accuracy2 = 0
            error = 0
            error_c = 0
            if retrain:
                with torch.no_grad():
                    for samples, labels in tqdm(train_loader, desc="Training"):
                        samples = samples.to(device)
                        labels = labels.to(device)

                        samples_hv = encode(samples)
                        model.add(samples_hv, labels)

            for iter in range(iterations):
                if method == "neural":
                    model.normalize()
                    idx = model.neural_regeneration(k=int(dimensions * 0.05))
                    encode.neural_regeneration(idx)
                    model.reset_parameters()

                for samples, labels in tqdm(
                    train_loader, desc="Training", disable=True
                ):
                    samples = samples.to(device)
                    labels = labels.to(device)

                    samples_hv = encode(samples)
                    if method == "add":
                        model.add(samples_hv, labels)
                    elif method == "add_online":
                        model.add_online(samples_hv, labels)
                    elif method == "add_adapt":
                        model.add_adapt(samples_hv, labels)
                    elif method == "add_adjust":
                        model.add_adjust(samples_hv, labels)
                    elif method == "add_adjust_2":
                        model.add_adjust_2(samples_hv, labels)
                    elif method == "add_adjust_3":
                        model.add_adjust_3(samples_hv, labels)
                    elif method == "neural":
                        model.add_adapt(samples_hv, labels)

                if iter in [5, 10, 15]:
                    with torch.no_grad():
                        m = copy.deepcopy(model)
                        m.normalize()

                        for samples, labels in tqdm(test_loader, desc="Testing"):
                            samples = samples.to(device)

                            samples_hv = encode(samples)
                            outputs = m(samples_hv, dot=True)
                            accuracy.update(outputs.cpu(), labels)

                    benchmark.report(dataset, accuracy.compute().item())
                    with open(results_file, "a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow(
                            [
                                dataset.name,
                                accuracy.compute().item(),
                                time.time() - t,
                                dimensions,
                                method,
                                encoding,
                                iter,
                                retrain,
                            ]
                        )

            with torch.no_grad():
                model.normalize()

                for samples, labels in tqdm(test_loader, desc="Testing"):
                    samples = samples.to(device)

                    samples_hv = encode(samples)
                    outputs = model(samples_hv, dot=True)

                    accuracy.update(outputs.cpu(), labels)
                    pred = torch.argmax(outputs)
                    # print(model.miss_predicted_large[pred])
                    outputs_large = torchhd.dot_similarity(
                        samples_hv, model.miss_predicted_large[pred]
                    )
                    # print(model.miss_predicted_large[pred])
                    # print(labels, pred, torch.argmax(outputs_large))
                    # print(torch.max(outputs), torch.max(outputs_large))
                    # print(num_train_samples/num_classes, model.miss_predicted_counter[pred][torch.argmax(outputs_large).item()])
                    # print(torch.argmax(outputs) != labels, torch.max(outputs), torchhd.dot_similarity(model.weight[torch.argmax(outputs_large)], samples_hv), torch.max(outputs_large))

                    if (
                        torch.max(outputs) < torch.max(outputs_large)
                        and int(int(num_train_samples / num_classes) * 0.2)
                        < torch.sum(model.miss_predicted_counter[pred])
                        and model.miss_predicted_counter[pred][
                            torch.argmax(outputs_large).item()
                        ]
                        > torch.sum(model.miss_predicted_counter[pred]) * 0.35
                    ):
                        pred = torch.argmax(outputs_large)
                    if pred == labels:
                        accuracy2 += 1
                        # print('CORRECT', torch.max(outputs), torch.max(outputs_large), abs(torch.max(outputs) - torch.max(outputs_large)))

                    else:
                        # print('INCORRECT',torch.max(outputs), torch.max(outputs_large), torch.argmax(outputs), torch.argmax(outputs), labels)
                        error += abs(torch.max(outputs) - torch.max(outputs_large))
                        error_c += 1
                    # print(labels, pred, pred == labels)
                    # print()

                    # if torch.max(outputs) > torchhd.dot_similarity(samples_hv,outputs_misspredict[torch.argmax(outputs)]):
                    #    accuracy2.update(outputs.cpu(), labels)
                    # else:
                    #    accuracy2.update(outputs_misspredict.cpu(), labels)

                    # print(torch.max(outputs), torch.max(outputs_misspredict), torch.max(torchhd.dot_similarity(samples_hv, model.miss_predicted_large[torch.argmax(outputs)][0])))

            print(accuracy.compute().item(), accuracy2 / len(test_loader))

            benchmark.report(dataset, accuracy.compute().item())
            with open(results_file, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        dataset.name,
                        accuracy.compute().item(),
                        accuracy2 / len(test_loader),
                        time.time() - t,
                        dimensions,
                        method,
                        encoding,
                        iterations,
                        retrain,
                    ]
                )
            # print(f"{dataset.name} accuracy: {(accuracy.compute().item() * 100):.2f}%")

    # Returns a dictionary with names of the datasets and their respective accuracy that is averaged over folds (if applicable) and repeats
    # benchmark_accuracy = benchmark.score()

    # print(benchmark_accuracy)


BATCH_SIZE = 1
# Specifies how many random initializations of the model to evaluate for each dataset in the collection.
REPEATS = 1
# DIMENSIONS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 10000]
DIMENSIONS = [10000]

# ENCODINGS = ["bundle", "sequence", "ngram", "hashmap", "flocet", "density", "random", "sinusoid"]
ENCODINGS = ["flocet"]
# ENCODINGS = ["sinusoid"]
# METHODS = ["add"]
METHODS = [
    "add",
    # "add_adapt",
    # "add_online",
    # "add_adjust",
    # "add_adjust_2",
    # "add_adjust_3",
]

# METHODS = ["neural"]
RETRAIN = [False]

ITERATIONS = 2

print(benchmark.datasets())

for i in DIMENSIONS:
    for j in ENCODINGS:
        for k in METHODS:
            for r in RETRAIN:
                exec_arena(
                    encoding=j,
                    method=k,
                    dimensions=i,
                    repeats=REPEATS,
                    retrain=r,
                    batch_size=BATCH_SIZE,
                    iterations=ITERATIONS,
                )
import pandas as pd

df = pd.read_csv(results_file)
var = "Method"
acc = "Accuracy"
methods_order = [
    "add",
]
df_mean = df.groupby([var, "Name"])[acc].mean().to_frame()
df_pivot = df_mean.reset_index().pivot(index="Name", columns=var, values=acc)
df_pivot.loc["mean"] = df_pivot.mean(axis=0)
df_pivot = df_pivot[methods_order].round(3)
# Print the new DataFrame to the console

print(df_pivot)
var = "Method"
acc = "AccuracyMiss"
methods_order = [
    "add",
]
df_mean = df.groupby([var, "Name"])[acc].mean().to_frame()
df_pivot = df_mean.reset_index().pivot(index="Name", columns=var, values=acc)
df_pivot.loc["mean"] = df_pivot.mean(axis=0)
df_pivot = df_pivot[methods_order].round(3)
print(df_pivot)
