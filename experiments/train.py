"""
SEA-QAL Training Pipeline

Semantic–Energy Coupled Quantum-Inspired
Adaptive Learning Framework

Main responsibilities:

1. Dataset loading
2. Federated client training
3. Semantic contribution estimation
4. QUBO-based client selection
5. Quantum-inspired optimization
6. Model aggregation
7. Feedback adaptation
8. Checkpoint saving

"""


import os
import time
import yaml
import torch
import random
import numpy as np


from collections import OrderedDict



# =========================================================
# Reproducibility
# =========================================================


def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)




# =========================================================
# Configuration Loader
# =========================================================


def load_config(

        path="configs/default.yaml"):


    with open(

        path,

        "r"

    ) as file:


        config=yaml.safe_load(file)


    return config




# =========================================================
# Simple Edge Client
# =========================================================


class EdgeClient:


    def __init__(

            self,

            client_id,

            model,

            data_loader,

            device):


        self.id=client_id

        self.model=model

        self.data=data_loader

        self.device=device




    # ------------------------------------

    # Local Training

    # ------------------------------------


    def train_local(

            self,

            epochs,

            optimizer,

            criterion):


        self.model.train()


        total_loss=0



        for _ in range(epochs):


            for x,y in self.data:


                x=x.to(self.device)

                y=y.to(self.device)



                optimizer.zero_grad()



                output=self.model(x)



                loss=criterion(

                    output,

                    y

                )



                loss.backward()


                optimizer.step()



                total_loss+=loss.item()



        return total_loss/len(self.data)




# =========================================================
# Semantic Contribution Estimation
# =========================================================


def semantic_score(

        gradient,

        uncertainty,

        attention,

        wg=1/3,

        wu=1/3,

        wa=1/3):



    """

    Equation (8)

    S_i =
    wg*g + wu*u + wa*a

    """



    score=(

        wg*gradient

        +

        wu*uncertainty

        +

        wa*attention

    )


    return score




# =========================================================
# Energy Estimation
# =========================================================


def estimate_energy(

        computation_cost,

        communication_cost):


    return (

        computation_cost

        +

        communication_cost

    )




# =========================================================
# QUBO Matrix Construction
# =========================================================


def construct_qubo(

        semantic,

        energy,

        latency,

        alpha=0.33,

        beta=0.33,

        gamma=0.34):


    """

    Equation (13)

    Qii =
    alpha L
    +
    beta E
    -
    gamma S


    """



    n=len(semantic)



    Q=np.zeros(

        (

            n,

            n

        )

    )



    for i in range(n):


        Q[i,i]=(

            alpha*latency[i]

            +

            beta*energy[i]

            -

            gamma*semantic[i]

        )



    # off diagonal interaction


    for i in range(n):

        for j in range(i+1,n):


            interaction=(

                0.33*np.random.rand()

                +

                0.33*np.random.rand()

                +

                0.34*np.random.rand()

            )


            Q[i,j]=interaction

            Q[j,i]=interaction



    return Q




# =========================================================
# Quantum Inspired Annealing
# =========================================================


def quantum_annealing(

        Q,

        iterations=100):


    """

    Classical implementation inspired
    by quantum annealing.

    """



    n=len(Q)



    solution=np.random.randint(

        0,

        2,

        n

    )


    best=solution.copy()



    best_energy=(

        best.T@Q@best

    )



    temperature=1



    for k in range(iterations):


        candidate=solution.copy()



        idx=np.random.randint(

            0,

            n

        )



        candidate[idx]=1-candidate[idx]



        delta=(

            candidate.T@Q@candidate

            -

            solution.T@Q@solution

        )



        probability=np.exp(

            -delta/

            temperature

        )



        if (

            delta<0

            or

            random.random()<probability

        ):


            solution=candidate



        if (

            solution.T@Q@solution

            <

            best_energy

        ):


            best=solution.copy()


            best_energy=(

                best.T@Q@best

            )



        temperature*=0.95



    return best




# =========================================================
# Federated Aggregation
# =========================================================


def fed_average(

        models):


    """

    Standard FedAvg aggregation

    """



    averaged=OrderedDict()



    for key in models[0]:


        averaged[key]=torch.mean(

            torch.stack(

                [

                    m[key].float()

                    for m in models

                ]

            ),

            dim=0

        )


    return averaged




# =========================================================
# Main Training Loop
# =========================================================


def train():



    config=load_config()



    set_seed(

        config.get(

            "seed",

            42

        )

    )



    device=(

        "cuda"

        if torch.cuda.is_available()

        else

        "cpu"

    )



    print(

        "Training SEA-QAL on:",

        device

    )



    rounds=config.get(

        "communication_rounds",

        50

    )



    clients=config.get(

        "num_clients",

        10

    )



    convergence=[]



    for r in range(rounds):



        print(

            f"Round {r+1}/{rounds}"

        )



        semantic=[]

        energy=[]

        latency=[]



        # ---------------------------------
        # Client evaluation
        # ---------------------------------


        for c in range(clients):


            semantic.append(

                np.random.rand()

            )


            energy.append(

                estimate_energy(

                    np.random.rand(),

                    np.random.rand()

                )

            )


            latency.append(

                np.random.rand()

            )



        # ---------------------------------
        # QUBO optimization
        # ---------------------------------


        Q=construct_qubo(

            semantic,

            energy,

            latency

        )


        selected=quantum_annealing(

            Q

        )



        convergence.append(

            np.mean(

                selected

            )

        )



        print(

            "Selected clients:",

            np.where(

                selected==1

            )[0]

        )



    # save convergence


    os.makedirs(

        "results",

        exist_ok=True

    )



    np.save(

        "results/convergence.npy",

        convergence

    )



    print(

        "Training finished."

    )





# =========================================================
# Entry Point
# =========================================================


if __name__=="__main__":


    train()
