"""
SEA-QAL Scalability Visualization Module


Generate scalability plots for increasing
edge-node configurations.


Corresponding to:

Section 4.8 Analysis of Scalability

Table 20:
Scalability assessment of SEA-QAL


Metrics:

- QUBO construction time
- Solver runtime
- Memory consumption
- Convergence epochs


"""


import os
import pandas as pd
import matplotlib.pyplot as plt




# =========================================================
# Configuration
# =========================================================


INPUT_FILE = "../results/scalability.csv"


OUTPUT_DIR = "../figures/scalability"



METRICS = [

    "QUBO Construction Time",

    "Solver Runtime",

    "Memory Consumption",

    "Convergence Epochs"

]





# =========================================================
# Load Scalability Results
# =========================================================


def load_data():


    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(

            f"Missing file: {INPUT_FILE}"

        )


    data=pd.read_csv(

        INPUT_FILE

    )


    return data





# =========================================================
# Generic Scalability Plot
# =========================================================


def plot_scalability_metric(

        data,

        metric):


    """

    Plot metric variation
    versus edge-node density.


    """


    plt.figure(

        figsize=(8,5)

    )



    plt.plot(

        data["Edge Nodes"],

        data[metric],

        marker="o",

        linewidth=2

    )



    plt.xlabel(

        "Number of Edge Nodes",

        fontsize=12

    )


    plt.ylabel(

        metric,

        fontsize=12

    )


    plt.title(

        f"SEA-QAL Scalability: {metric}",

        fontsize=13

    )


    plt.grid(

        True,

        linestyle="--",

        alpha=0.4

    )


    plt.tight_layout()



    os.makedirs(

        OUTPUT_DIR,

        exist_ok=True

    )



    filename=os.path.join(

        OUTPUT_DIR,

        metric.lower()

        .replace(

            " ",

            "_"

        )

        +

        ".png"

    )



    plt.savefig(

        filename,

        dpi=600,

        bbox_inches="tight"

    )


    plt.close()



    print(

        "Saved:",

        filename

    )





# =========================================================
# Combined Computational Growth Plot
# =========================================================


def plot_runtime_growth(

        data):


    """

    Compare computational growth
    of QUBO construction and solver.


    """


    plt.figure(

        figsize=(8,5)

    )


    plt.plot(

        data["Edge Nodes"],

        data["QUBO Construction Time"],

        marker="o",

        linewidth=2,

        label="QUBO Construction"

    )


    plt.plot(

        data["Edge Nodes"],

        data["Solver Runtime"],

        marker="s",

        linewidth=2,

        label="Solver Runtime"

    )



    plt.xlabel(

        "Number of Edge Nodes",

        fontsize=12

    )


    plt.ylabel(

        "Execution Time (s)",

        fontsize=12

    )


    plt.title(

        "Computational Scalability of SEA-QAL",

        fontsize=13

    )


    plt.legend()



    plt.grid(

        True,

        linestyle="--",

        alpha=0.4

    )


    plt.tight_layout()



    filename=os.path.join(

        OUTPUT_DIR,

        "computational_scalability.png"

    )



    plt.savefig(

        filename,

        dpi=600,

        bbox_inches="tight"

    )


    plt.close()



    print(

        "Saved:",

        filename

    )





# =========================================================
# QUBO Matrix Growth Visualization
# =========================================================


def plot_qubo_growth(

        data):


    """

    Demonstrate quadratic growth
    of QUBO matrix size.


    QUBO size ≈ N^2


    """


    plt.figure(

        figsize=(8,5)

    )


    plt.plot(

        data["Edge Nodes"],

        data["QUBO Size"],

        marker="o",

        linewidth=2

    )


    plt.xlabel(

        "Number of Edge Nodes",

        fontsize=12

    )


    plt.ylabel(

        "QUBO Matrix Size",

        fontsize=12

    )


    plt.title(

        "Quadratic Growth of QUBO Representation",

        fontsize=13

    )


    plt.grid(

        True,

        linestyle="--",

        alpha=0.4

    )



    plt.tight_layout()



    filename=os.path.join(

        OUTPUT_DIR,

        "qubo_growth.png"

    )



    plt.savefig(

        filename,

        dpi=600,

        bbox_inches="tight"

    )


    plt.close()



    print(

        "Saved:",

        filename

    )





# =========================================================
# Main
# =========================================================


def main():


    data=load_data()



    for metric in METRICS:


        plot_scalability_metric(

            data,

            metric

        )



    plot_runtime_growth(

        data

    )



    plot_qubo_growth(

        data

    )




if __name__=="__main__":


    main()
