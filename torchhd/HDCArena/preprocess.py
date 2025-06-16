import torch
from torch import Tensor
import torch.utils.data as data
from torchhd.datasets import EuropeanLanguages as Languages
from torchvision import transforms
import PIL
from torch.utils.data import random_split, TensorDataset


def create_min_max_normalize(min: Tensor, max: Tensor, device):
    def normalize(input: Tensor) -> Tensor:
        if type(input) == PIL.Image.Image:
            convert_tensor = transforms.ToTensor()
            return torch.nan_to_num(
                (convert_tensor(input).view(32, 32, 3) - min) / (max - min)
            )
        return torch.nan_to_num((torch.tensor(input).to(device) - min) / (max - min))

    return normalize


def preprocess(dataset, batch_size, device, partial_data, method):
    if dataset.name == "EuropeanLanguages":
        MAX_INPUT_SIZE = 128
        PADDING_IDX = 0

        ASCII_A = ord("a")
        ASCII_Z = ord("z")
        ASCII_SPACE = ord(" ")

        def char2int(char: str) -> int:
            """Map a character to its integer identifier"""
            ascii_index = ord(char)

            if ascii_index == ASCII_SPACE:
                return ASCII_Z - ASCII_A + 1
            return ascii_index - ASCII_A

        def transform(x: str) -> torch.Tensor:
            char_ids = x[:MAX_INPUT_SIZE]
            char_ids = [char2int(char) + 1 for char in char_ids.lower()]

            if len(char_ids) < MAX_INPUT_SIZE:
                char_ids += [PADDING_IDX] * (MAX_INPUT_SIZE - len(char_ids))

            return torch.tensor(char_ids, dtype=torch.long)

        num_feat = MAX_INPUT_SIZE
        train_ds = Languages("../data", train=True, transform=transform, download=True)
        test_ds = Languages("../data", train=False, transform=transform, download=True)
        num_classes = len(train_ds.classes)
        partial_data = int(partial_data * len(train_ds))
        train_ds = torch.utils.data.random_split(
            train_ds, [partial_data, len(train_ds) - partial_data]
        )[0]
    elif dataset.name in ["PAMAP", "EMGHandGestures"]:
        if dataset.name == "EMGHandGestures":
            num_feat = dataset.train[0][0].size(-1) * dataset.train[0][0].size(-2)
        else:
            num_feat = dataset.train[0][0].size(-1)

        num_classes = len(dataset.train.classes)

        min_val = torch.min(dataset.train.data, 0).values.to(device)
        max_val = torch.max(dataset.train.data, 0).values.to(device)
        transform = create_min_max_normalize(min_val, max_val, device)
        dataset.train.transform = transform

        train_size = int(len(dataset.train) * 0.7)
        test_size = len(dataset.train) - train_size
        train_ds, test_ds = data.random_split(dataset.train, [train_size, test_size])
        partial_data = int(partial_data * len(train_ds))
        train_ds = torch.utils.data.random_split(
            train_ds, [partial_data, len(train_ds) - partial_data]
        )[0]
    else:
        # Number of features in the dataset.
        if dataset.name not in ["MNIST", "CIFAR10"]:
            num_feat = dataset.train[0][0].size(-1)
        else:
            if dataset.name == "MNIST":
                num_feat = dataset.train[0][0].size(-1) * dataset.train[0][0].size(-1)
            elif dataset.name == "CIFAR10":
                num_feat = 3072
        num_classes = len(dataset.train.classes)

        if dataset.name not in ["MNIST", "CIFAR10"]:
            min_val = torch.min(dataset.train.data, 0).values.to(device)
            max_val = torch.max(dataset.train.data, 0).values.to(device)
            transform_train = create_min_max_normalize(min_val, max_val, device)
            dataset.train.transform = transform_train
            min_val = torch.min(dataset.test.data, 0).values.to(device)
            max_val = torch.max(dataset.test.data, 0).values.to(device)
            transform_test = create_min_max_normalize(min_val, max_val, device)
            dataset.test.transform = transform_test

            if method == "rvfl":
                min_val = torch.min(dataset.train.data, 0).values.to(device)
                max_val = torch.max(dataset.train.data, 0).values.to(device)
                transform = create_min_max_normalize(min_val, max_val, device)
                dataset.train.transform = transform
                dataset.test.transform = transform
                train_ds = transform(dataset.train.data)
                partial_data = int(partial_data * len(train_ds))
                train_ds = train_ds[:partial_data]
            else:
                train_ds = dataset.train
                partial_data = int(partial_data * len(train_ds))
                if partial_data == 0:
                    partial_data = 1
                train_ds = torch.utils.data.random_split(
                    train_ds, [partial_data, len(train_ds) - partial_data]
                )[0]
        else:
            if method == "rvfl":
                min_val = torch.min(torch.tensor(dataset.train.data), 0).values.to(
                    device
                )
                max_val = torch.max(torch.tensor(dataset.train.data), 0).values.to(
                    device
                )
                transform = create_min_max_normalize(min_val, max_val, device)
                dataset.train.transform = transform
                dataset.test.transform = transform
                train_ds = transform(dataset.train.data)
                partial_data = int(partial_data * len(train_ds))
                train_ds = train_ds[:partial_data]
            else:
                train_ds = dataset.train
                partial_data = int(partial_data * len(train_ds))
                train_ds = torch.utils.data.random_split(
                    train_ds, [partial_data, len(train_ds) - partial_data]
                )[0]

        test_ds = dataset.test

    train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_ds, batch_size=batch_size)
    return train_loader, test_loader, num_classes, num_feat, train_ds, test_ds
