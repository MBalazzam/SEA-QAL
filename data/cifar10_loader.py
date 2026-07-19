"""
SEA-QAL CIFAR-10 Dataset Loader

This module provides:
- CIFAR-10 loading
- Data preprocessing
- IID and Non-IID partitioning for cloud-edge federated experiments
- Reproducible edge client data allocation

Author:
SEA-QAL Reproducibility Repository
"""

import random
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from torchvision import datasets, transforms


# ==========================================================
# Reproducibility
# ==========================================================

def set_seed(seed=42):
    """
    Set random seeds for reproducible experiments.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# ==========================================================
# CIFAR-10 Transform
# ==========================================================

def get_transform():

    return transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )

    ])



# ==========================================================
# Load CIFAR-10 Dataset
# ==========================================================

def load_cifar10(
        data_path="./data",
        download=True):

    """
    Load CIFAR-10 training and testing datasets.

    Returns:
        train_dataset
        test_dataset
    """

    transform = get_transform()


    train_dataset = datasets.CIFAR10(
        root=data_path,
        train=True,
        download=download,
        transform=transform
    )


    test_dataset = datasets.CIFAR10(
        root=data_path,
        train=False,
        download=download,
        transform=transform
    )


    return train_dataset, test_dataset



# ==========================================================
# IID Partition
# ==========================================================

def iid_partition(
        dataset,
        num_clients,
        seed=42):

    """
    IID partitioning of dataset among edge nodes.
    """

    set_seed(seed)


    dataset_size = len(dataset)

    indices = list(range(dataset_size))

    random.shuffle(indices)


    split_size = dataset_size // num_clients


    client_indices = {}


    for i in range(num_clients):

        start = i * split_size


        if i == num_clients - 1:

            end = dataset_size

        else:

            end = start + split_size


        client_indices[i] = indices[start:end]


    return client_indices



# ==========================================================
# Non-IID Partition using Dirichlet Distribution
# ==========================================================

def noniid_partition(
        dataset,
        num_clients,
        alpha=0.5,
        seed=42):

    """
    Generate heterogeneous Non-IID edge clients.

    Dirichlet distribution is used to simulate
    heterogeneous data availability.

    Args:
        alpha:
            Smaller values create stronger heterogeneity.

    """

    set_seed(seed)


    labels = np.array(dataset.targets)


    num_classes = len(np.unique(labels))


    client_indices = {
        i: []
        for i in range(num_clients)
    }


    for c in range(num_classes):

        class_indices = np.where(
            labels == c
        )[0]


        np.random.shuffle(class_indices)


        proportions = np.random.dirichlet(
            np.repeat(alpha, num_clients)
        )


        proportions = (
            np.cumsum(proportions)
            * len(class_indices)
        ).astype(int)


        split_indices = np.split(
            class_indices,
            proportions[:-1]
        )


        for client_id, idx in enumerate(split_indices):

            client_indices[client_id].extend(
                idx.tolist()
            )


    return client_indices



# ==========================================================
# Create Federated DataLoaders
# ==========================================================

def create_client_loaders(
        dataset,
        client_indices,
        batch_size=64):

    """
    Generate DataLoader for each edge client.
    """


    client_loaders = {}


    for client_id, indices in client_indices.items():


        subset = Subset(
            dataset,
            indices
        )


        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )


        client_loaders[client_id] = loader


    return client_loaders



# ==========================================================
# Complete CIFAR-10 Federated Loader
# ==========================================================

def get_cifar10_federated_loaders(
        num_clients=10,
        batch_size=64,
        iid=False,
        alpha=0.5,
        seed=42,
        data_path="./data"):

    """
    Main loader used by SEA-QAL experiments.

    Returns:

        client_loaders:
            Federated edge-node loaders

        test_loader:
            Global evaluation loader

    """


    set_seed(seed)


    train_dataset, test_dataset = load_cifar10(
        data_path=data_path
    )


    if iid:

        partitions = iid_partition(
            train_dataset,
            num_clients,
            seed
        )

    else:

        partitions = noniid_partition(
            train_dataset,
            num_clients,
            alpha,
            seed
        )


    client_loaders = create_client_loaders(
        train_dataset,
        partitions,
        batch_size
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )


    return client_loaders, test_loader



# ==========================================================
# Example Execution
# ==========================================================

if __name__ == "__main__":


    clients, test_loader = get_cifar10_federated_loaders(
        num_clients=10,
        batch_size=64,
        iid=False,
        alpha=0.5
    )


    print(
        "Number of edge clients:",
        len(clients)
    )


    for cid, loader in clients.items():

        x, y = next(iter(loader))

        print(
            f"Client {cid}: "
            f"Batch shape {x.shape}"
        )

        break
