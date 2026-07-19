"""
SEA-QAL Edge-IIoTset Dataset Loader

This module provides:
- Edge-IIoTset loading
- Data preprocessing
- Feature normalization
- IID / Non-IID edge client partitioning
- Federated DataLoaders

Dataset:
Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset
for IoT and IIoT Applications

"""

import os
import random
import numpy as np
import pandas as pd


import torch

from torch.utils.data import (
    Dataset,
    DataLoader,
    Subset
)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


# ==========================================================
# Reproducibility
# ==========================================================


def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# ==========================================================
# Custom Dataset
# ==========================================================


class EdgeIIoTDataset(Dataset):

    """
    PyTorch wrapper for Edge-IIoTset.
    """


    def __init__(self, features, labels):

        self.features = torch.tensor(
            features,
            dtype=torch.float32
        )


        self.labels = torch.tensor(
            labels,
            dtype=torch.long
        )


    def __len__(self):

        return len(self.labels)



    def __getitem__(self, index):

        return (
            self.features[index],
            self.labels[index]
        )



# ==========================================================
# Load and preprocess Edge-IIoTset
# ==========================================================


def load_edgeiiotset(
        csv_path,
        label_column="Attack_type"):


    """
    Load Edge-IIoTset CSV file.

    Parameters
    ----------
    csv_path:
        Path to Edge-IIoTset csv file

    label_column:
        Target attack/class label


    Returns
    -------

    Dataset object

    """


    if not os.path.exists(csv_path):

        raise FileNotFoundError(
            f"Dataset not found: {csv_path}"
        )



    df = pd.read_csv(csv_path)



    # Remove duplicated samples

    df = df.drop_duplicates()



    # Replace infinite values

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )


    # Fill missing values

    df.fillna(
        df.median(
            numeric_only=True
        ),
        inplace=True
    )



    # Separate label

    if label_column not in df.columns:

        raise ValueError(
            f"Label column {label_column} not found"
        )


    labels = df[label_column]


    features = df.drop(
        columns=[label_column]
    )



    # Remove categorical non-feature columns


    categorical_columns = features.select_dtypes(
        include=["object"]
    ).columns



    features = features.drop(
        columns=categorical_columns
    )



    # Convert to numpy


    X = features.values


    y = labels.values



    # Encode labels


    encoder = LabelEncoder()

    y = encoder.fit_transform(y)



    # Normalize features


    scaler = MinMaxScaler()

    X = scaler.fit_transform(X)



    dataset = EdgeIIoTDataset(
        X,
        y
    )


    return dataset, encoder, scaler




# ==========================================================
# IID Partition
# ==========================================================


def iid_partition(
        dataset,
        num_clients,
        seed=42):


    set_seed(seed)



    indices = list(
        range(len(dataset))
    )


    random.shuffle(indices)



    split_size = len(indices)//num_clients



    client_indices = {}



    for i in range(num_clients):


        start = i*split_size


        if i == num_clients-1:

            end=len(indices)

        else:

            end=start+split_size



        client_indices[i]=indices[start:end]



    return client_indices




# ==========================================================
# Non-IID Partition
# ==========================================================


def noniid_partition(
        dataset,
        num_clients,
        alpha=0.5,
        seed=42):


    """
    Dirichlet based heterogeneous partitioning.

    Suitable for federated edge scenarios.
    """


    set_seed(seed)



    labels = dataset.labels.numpy()



    classes = np.unique(labels)



    client_indices = {

        i:[]

        for i in range(num_clients)

    }



    for c in classes:


        class_indices = np.where(
            labels == c
        )[0]



        np.random.shuffle(
            class_indices
        )



        proportions = np.random.dirichlet(

            np.ones(num_clients)*alpha

        )



        split_points = (

            np.cumsum(proportions)
            *
            len(class_indices)

        ).astype(int)



        splits=np.split(
            class_indices,
            split_points[:-1]
        )



        for client_id, idx in enumerate(splits):


            client_indices[client_id].extend(
                idx.tolist()
            )



    return client_indices




# ==========================================================
# Create Edge DataLoaders
# ==========================================================


def create_client_loaders(
        dataset,
        partitions,
        batch_size=64):


    loaders={}



    for client_id, indices in partitions.items():


        subset = Subset(
            dataset,
            indices
        )


        loaders[client_id]=DataLoader(

            subset,

            batch_size=batch_size,

            shuffle=True,

            drop_last=False

        )


    return loaders




# ==========================================================
# Main Federated Edge-IIoT Loader
# ==========================================================


def get_edgeiiot_federated_loaders(

        csv_path,

        num_clients=10,

        batch_size=64,

        iid=False,

        alpha=0.5,

        seed=42):


    """
    Complete loader used in SEA-QAL experiments.
    """


    set_seed(seed)



    dataset, encoder, scaler = load_edgeiiotset(
        csv_path
    )



    if iid:


        partitions=iid_partition(

            dataset,

            num_clients,

            seed

        )

    else:


        partitions=noniid_partition(

            dataset,

            num_clients,

            alpha,

            seed

        )



    client_loaders=create_client_loaders(

        dataset,

        partitions,

        batch_size

    )



    return (

        client_loaders,

        encoder,

        scaler

    )




# ==========================================================
# Example
# ==========================================================


if __name__=="__main__":


    clients, encoder, scaler = get_edgeiiot_federated_loaders(

        csv_path="./data/Edge-IIoTset.csv",

        num_clients=10,

        batch_size=128,

        iid=False,

        alpha=0.5

    )



    print(
        "Number of Edge Clients:",
        len(clients)
    )


    for cid, loader in clients.items():


        x,y=next(iter(loader))


        print(
            f"Client {cid}:",
            x.shape,
            y.shape
        )


        break
