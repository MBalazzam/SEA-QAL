"""
SEA-QAL Synthetic Edge Dataset Generator

This module generates controlled synthetic cloud-edge workloads
for robustness and sensitivity analysis.

Generated attributes:
- semantic contribution
- gradient sensitivity
- prediction uncertainty
- attention concentration
- computation demand
- communication load
- energy consumption
- latency
- workload label

The generated dataset is used only for controlled analysis,
not as a replacement for real-world benchmarks.
"""


import os
import random
import numpy as np
import pandas as pd



# ==========================================================
# Reproducibility
# ==========================================================


def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)



# ==========================================================
# Semantic Attribute Generation
# ==========================================================


def generate_semantic_features(
        num_samples):


    """
    Generate semantic contribution indicators.

    Values are normalized in [0,1].
    """


    gradient_sensitivity = np.random.beta(
        2,
        5,
        num_samples
    )


    uncertainty = np.random.beta(
        3,
        4,
        num_samples
    )


    attention = np.random.beta(
        4,
        3,
        num_samples
    )


    semantic_score = (

        gradient_sensitivity
        +
        uncertainty
        +
        attention

    ) / 3



    return (

        gradient_sensitivity,

        uncertainty,

        attention,

        semantic_score

    )



# ==========================================================
# Edge Resource Generation
# ==========================================================


def generate_resource_features(
        num_samples):


    """
    Generate heterogeneous edge resource states.
    """


    cpu_capacity=np.random.uniform(

        1.2,

        2.4,

        num_samples

    )


    memory_capacity=np.random.uniform(

        4,

        16,

        num_samples

    )


    data_size=np.random.uniform(

        0.5,

        10,

        num_samples

    )


    bandwidth=np.random.uniform(

        5,

        50,

        num_samples

    )


    return (

        cpu_capacity,

        memory_capacity,

        data_size,

        bandwidth

    )



# ==========================================================
# Energy and Latency Model
# ==========================================================


def compute_energy_latency(

        cpu,

        data_size,

        bandwidth,

        semantic_score):


    """
    Controlled edge energy model.

    Higher computation and communication
    increase cost.

    """


    computation_energy = (

        cpu ** 2

        *

        np.random.uniform(
            0.8,
            1.2,
            len(cpu)
        )

    )


    communication_energy=(

        data_size /

        bandwidth

    )



    total_energy=(

        computation_energy

        +

        communication_energy

    )



    latency=(

        data_size/bandwidth

        +

        1/cpu

    )



    # semantic-aware correction

    latency = latency * (

        1 -

        0.2*semantic_score

    )



    return (

        total_energy,

        latency

    )



# ==========================================================
# Workload Label Generation
# ==========================================================


def generate_labels(

        semantic_score,

        energy):


    """

    Labels:

    0 : Low semantic workload

    1 : Medium semantic workload

    2 : High semantic workload


    """


    labels=np.zeros(
        len(semantic_score)
    )


    for i in range(
        len(semantic_score)
    ):


        if semantic_score[i] > 0.65:

            labels[i]=2


        elif semantic_score[i] > 0.35:

            labels[i]=1


        else:

            labels[i]=0



    return labels.astype(int)



# ==========================================================
# Synthetic Dataset Generator
# ==========================================================


def generate_synthetic_edge_dataset(

        num_samples=10000,

        num_clients=20,

        output_path=

        "./synthetic_edge_dataset.csv",

        seed=42):


    """

    Generate synthetic cloud-edge workload dataset.


    """


    set_seed(seed)



    # semantic attributes


    (

        gradient,

        uncertainty,

        attention,

        semantic

    ) = generate_semantic_features(

        num_samples

    )



    # resource attributes


    (

        cpu,

        memory,

        data_size,

        bandwidth

    ) = generate_resource_features(

        num_samples

    )



    # energy and latency


    (

        energy,

        latency

    ) = compute_energy_latency(

        cpu,

        data_size,

        bandwidth,

        semantic

    )



    labels=generate_labels(

        semantic,

        energy

    )



    # Edge node assignment

    edge_nodes=np.random.randint(

        0,

        num_clients,

        num_samples

    )



    # Dataset dataframe


    df=pd.DataFrame({


        "gradient_sensitivity":

        gradient,


        "prediction_uncertainty":

        uncertainty,


        "attention_concentration":

        attention,


        "semantic_score":

        semantic,


        "cpu_capacity_GHz":

        cpu,


        "memory_GB":

        memory,


        "data_size_MB":

        data_size,


        "bandwidth_Mbps":

        bandwidth,


        "energy_consumption":

        energy,


        "latency_ms":

        latency*1000,


        "edge_node":

        edge_nodes,


        "workload_class":

        labels


    })



    # Save


    os.makedirs(

        os.path.dirname(output_path)

        if os.path.dirname(output_path)

        else ".",

        exist_ok=True

    )


    df.to_csv(

        output_path,

        index=False

    )



    return df



# ==========================================================
# Non-IID Synthetic Partition
# ==========================================================


def create_non_iid_clients(

        dataframe,

        num_clients=20):


    """

    Create heterogeneous edge workloads.

    """


    clients={}



    for i in range(num_clients):


        clients[i]=dataframe[

            dataframe["edge_node"]

            == i

        ]



    return clients



# ==========================================================
# Main Execution
# ==========================================================


if __name__=="__main__":


    dataset=generate_synthetic_edge_dataset(

        num_samples=10000,

        num_clients=20,

        output_path=

        "./synthetic_edge_dataset.csv",

        seed=42

    )



    print(
        "Synthetic dataset generated:"
    )


    print(
        dataset.head()
    )


    print(
        "\nDataset shape:",
        dataset.shape
    )
