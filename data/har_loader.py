"""
SEA-QAL HAR Dataset Loader

Human Activity Recognition Using Smartphones Dataset

Features:
- IMU sensor feature loading
- preprocessing
- normalization
- federated client partitioning
- IID / Non-IID split

Dataset:
UCI Human Activity Recognition Using Smartphones

"""


import os
import random
import numpy as np

import torch

from torch.utils.data import (
    Dataset,
    DataLoader,
    Subset
)

from sklearn.preprocessing import StandardScaler



# ==========================================================
# Reproducibility
# ==========================================================


def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic=True

    torch.backends.cudnn.benchmark=False



# ==========================================================
# HAR Dataset Class
# ==========================================================


class HARDataset(Dataset):


    def __init__(
            self,
            features,
            labels):


        self.features=torch.tensor(
            features,
            dtype=torch.float32
        )


        self.labels=torch.tensor(
            labels,
            dtype=torch.long
        )


    def __len__(self):

        return len(self.labels)



    def __getitem__(
            self,
            index):


        return (

            self.features[index],

            self.labels[index]

        )



# ==========================================================
# Load HAR Dataset
# ==========================================================


def load_har_dataset(
        root_dir):


    """
    Load UCI HAR dataset.

    Directory:

    root_dir/
        train/
            X_train.txt
            y_train.txt

        test/
            X_test.txt
            y_test.txt


    Returns:
        PyTorch Dataset

    """



    train_path=os.path.join(
        root_dir,
        "train"
    )


    test_path=os.path.join(
        root_dir,
        "test"
    )



    if not os.path.exists(train_path):

        raise FileNotFoundError(
            "HAR train directory not found"
        )



    # -----------------------------
    # Load features
    # -----------------------------


    X_train=np.loadtxt(

        os.path.join(
            train_path,
            "X_train.txt"
        )

    )


    X_test=np.loadtxt(

        os.path.join(
            test_path,
            "X_test.txt"
        )

    )



    # -----------------------------
    # Load labels
    # -----------------------------


    y_train=np.loadtxt(

        os.path.join(
            train_path,
            "y_train.txt"
        )

    )


    y_test=np.loadtxt(

        os.path.join(
            test_path,
            "y_test.txt"
        )

    )



    # Merge train/test


    X=np.concatenate(

        [
            X_train,
            X_test
        ],

        axis=0

    )


    y=np.concatenate(

        [
            y_train,
            y_test
        ]

    )



    # Convert labels
    # Original labels: 1-6

    y=y.astype(int)-1



    # Feature normalization


    scaler=StandardScaler()


    X=scaler.fit_transform(
        X
    )



    dataset=HARDataset(
        X,
        y
    )



    return dataset, scaler




# ==========================================================
# IID Federated Partition
# ==========================================================


def iid_partition(

        dataset,

        num_clients,

        seed=42):


    set_seed(seed)



    indices=list(
        range(len(dataset))
    )



    random.shuffle(
        indices
    )



    split=len(indices)//num_clients



    partitions={}



    for i in range(num_clients):


        start=i*split


        if i==num_clients-1:

            end=len(indices)

        else:

            end=start+split



        partitions[i]=indices[start:end]



    return partitions




# ==========================================================
# Non-IID Federated Partition
# ==========================================================


def noniid_partition(

        dataset,

        num_clients,

        alpha=0.5,

        seed=42):


    """
    Dirichlet distribution based
    heterogeneous sensor partition.

    """


    set_seed(seed)



    labels=dataset.labels.numpy()



    classes=np.unique(
        labels
    )



    client_indices={

        i:[]

        for i in range(num_clients)

    }



    for c in classes:


        class_samples=np.where(

            labels==c

        )[0]


        np.random.shuffle(
            class_samples
        )



        proportions=np.random.dirichlet(

            np.ones(num_clients)*alpha

        )



        boundaries=(

            np.cumsum(proportions)
            *
            len(class_samples)

        ).astype(int)



        splits=np.split(

            class_samples,

            boundaries[:-1]

        )



        for client_id, split in enumerate(splits):


            client_indices[client_id].extend(

                split.tolist()

            )



    return client_indices




# ==========================================================
# Create Client DataLoaders
# ==========================================================


def create_client_loaders(

        dataset,

        partitions,

        batch_size=64):



    loaders={}



    for client_id,indices in partitions.items():


        subset=Subset(

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
# Main HAR Federated Loader
# ==========================================================


def get_har_federated_loaders(

        root_dir,

        num_clients=10,

        batch_size=64,

        iid=False,

        alpha=0.5,

        seed=42):



    set_seed(seed)



    dataset, scaler=load_har_dataset(

        root_dir

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



    loaders=create_client_loaders(

        dataset,

        partitions,

        batch_size

    )



    return (

        loaders,

        scaler

    )




# ==========================================================
# Test
# ==========================================================


if __name__=="__main__":


    clients, scaler=get_har_federated_loaders(

        root_dir="./data/UCI_HAR",

        num_clients=10,

        batch_size=64,

        iid=False,

        alpha=0.5

    )



    print(
        "Number of clients:",
        len(clients)
    )


    for cid,loader in clients.items():


        X,y=next(iter(loader))


        print(
            "Client:",
            cid,
            "Feature shape:",
            X.shape,
            "Label shape:",
            y.shape
        )


        break
