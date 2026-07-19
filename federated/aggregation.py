"""
SEA-QAL Federated Aggregation Module

This module implements:

1. FedAvg aggregation
2. Semantic-aware weighted aggregation
3. Energy-aware aggregation
4. Adaptive SEA-QAL aggregation

Based on:

Federated learning aggregation:
    Section 3

Semantic contribution:
    Eq. (8)

Energy-aware optimization:
    Eq. (2)

"""


import torch
from collections import OrderedDict
import numpy as np




# =========================================================
# Basic FedAvg Aggregation
# =========================================================


def fedavg(

        client_models,

        client_weights=None):

    """
    Standard Federated Averaging.


    Parameters
    ----------

    client_models:
        list of model state dictionaries


    client_weights:
        number of samples per client


    Returns
    -------

    aggregated model


    Formula:

    W = Σ(n_i/n) W_i

    """



    if client_weights is None:


        client_weights=[

            1/len(client_models)

            for _ in client_models

        ]



    aggregated=OrderedDict()



    for key in client_models[0].keys():


        aggregated[key]=sum(

            [

                client_models[i][key].float()

                *

                client_weights[i]

                for i in range(

                    len(client_models)

                )

            ]

        )



    return aggregated




# =========================================================
# Normalize Weights
# =========================================================


def normalize_weights(

        weights):


    """

    Min-max normalization

    """



    weights=np.asarray(

        weights,

        dtype=float

    )


    if np.max(weights)==np.min(weights):


        return np.ones_like(

            weights

        )/len(weights)



    weights=(

        weights-np.min(weights)

    )/(

        np.max(weights)

        -

        np.min(weights)

    )



    weights=(

        weights/

        np.sum(weights)

    )


    return weights




# =========================================================
# Semantic-Energy Weighted Aggregation
# =========================================================


def semantic_energy_weighted_aggregation(

        client_models,

        semantic_scores,

        energy_costs,

        latency_values,

        alpha=0.33,

        beta=0.33,

        gamma=0.34):


    """

    SEA-QAL aggregation mechanism.



    Client contribution score:



    A_i =

    alpha*S_i

    + beta*(1/E_i)

    + gamma*(1/L_i)



    """


    semantic_scores=np.asarray(

        semantic_scores

    )


    energy_costs=np.asarray(

        energy_costs

    )


    latency_values=np.asarray(

        latency_values

    )



    # Energy efficiency


    energy_eff=1/(

        energy_costs+1e-8

    )



    # Latency efficiency


    latency_eff=1/(

        latency_values+1e-8

    )



    # Normalize


    semantic_norm=normalize_weights(

        semantic_scores

    )


    energy_norm=normalize_weights(

        energy_eff

    )


    latency_norm=normalize_weights(

        latency_eff

    )



    # Final aggregation score


    scores=(

        alpha*semantic_norm

        +

        beta*energy_norm

        +

        gamma*latency_norm

    )



    scores=normalize_weights(

        scores

    )



    aggregated=OrderedDict()



    for key in client_models[0]:


        aggregated[key]=sum(

            [

                client_models[i][key].float()

                *

                scores[i]

                for i in range(

                    len(client_models)

                )

            ]

        )



    return aggregated, scores




# =========================================================
# Selected Client Aggregation
# =========================================================


def aggregate_selected_clients(

        client_models,

        selected_clients,

        semantic_scores,

        energy_costs,

        latency_values):


    """

    Aggregate only clients selected
    by QUBO optimizer.



    """



    selected_models=[

        client_models[i]

        for i in selected_clients

    ]



    selected_semantic=[

        semantic_scores[i]

        for i in selected_clients

    ]


    selected_energy=[

        energy_costs[i]

        for i in selected_clients

    ]


    selected_latency=[

        latency_values[i]

        for i in selected_clients

    ]



    aggregated,weights=(

        semantic_energy_weighted_aggregation(

            selected_models,

            selected_semantic,

            selected_energy,

            selected_latency

        )

    )


    return aggregated,weights




# =========================================================
# Adaptive Feedback Aggregation
# =========================================================


def adaptive_feedback_aggregation(

        client_models,

        semantic_scores,

        energy_costs,

        latency_values,

        energy_budget,

        current_energy):


    """

    Feedback-controlled aggregation.



    If energy exceeds budget:

        increase energy importance



    Otherwise:

        maintain semantic priority



    """



    if current_energy > energy_budget:


        beta=0.55

        alpha=0.25

        gamma=0.20



    else:


        alpha=0.40

        beta=0.30

        gamma=0.30



    return semantic_energy_weighted_aggregation(

        client_models,

        semantic_scores,

        energy_costs,

        latency_values,

        alpha,

        beta,

        gamma

    )




# =========================================================
# Model Difference Utility
# =========================================================


def model_difference(

        global_model,

        local_model):


    """

    Compute local update difference.



    Used for:

    gradient sensitivity estimation


    """



    difference={}



    for key in global_model:


        difference[key]=(

            local_model[key]

            -

            global_model[key]

        )


    return difference




# =========================================================
# Aggregate Multiple Rounds
# =========================================================


def federated_training_aggregation(

        global_model,

        local_models,

        method="SEA-QAL",

        **kwargs):


    """

    High-level aggregation interface.



    """



    if method=="FedAvg":


        return fedavg(

            local_models

        )



    elif method=="SEA-QAL":


        aggregated,_=(

            semantic_energy_weighted_aggregation(

                local_models,

                kwargs["semantic_scores"],

                kwargs["energy_costs"],

                kwargs["latency_values"]

            )

        )


        return aggregated



    else:


        raise ValueError(

            "Unknown aggregation method"

        )




# =========================================================
# Test Example
# =========================================================


if __name__=="__main__":


    # Fake models


    model1={

        "layer":

        torch.tensor(

            [1.,2.,3.]

        )

    }


    model2={

        "layer":

        torch.tensor(

            [2.,3.,4.]

        )

    }



    models=[

        model1,

        model2

    ]



    result,weights=(

        semantic_energy_weighted_aggregation(

            models,

            semantic_scores=[0.9,0.7],

            energy_costs=[100,150],

            latency_values=[20,30]

        )

    )



    print(

        "Aggregation weights:",

        weights

    )


    print(

        "Aggregated model:",

        result

    )
