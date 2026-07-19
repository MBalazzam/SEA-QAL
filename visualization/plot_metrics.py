"""
SEA-QAL Performance Metrics Visualization


Generate comparison plots for:

1. Accuracy
2. Energy Consumption
3. Latency
4. Sustainability Index


Used for:
- Main performance comparison
- Dataset-wise analysis
- Sensitivity analysis


"""


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np




# =========================================================
# Configuration
# =========================================================


INPUT_FILE = "../results/metrics.csv"


OUTPUT_DIR = "../figures/metrics"



METRICS = [

    "Accuracy",

    "Energy",

    "Latency",

    "Sustainability"

]



# =========================================================
# Load Results
# =========================================================


def load_results():

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(

            f"Missing file: {INPUT_FILE}"

        )


    data=pd.read_csv(

        INPUT_FILE

    )


    return data




# =========================================================
# Bar Plot
# =========================================================


def plot_metric(

        data,

        metric):


    """

    Generate comparison bar chart.


    CSV format:

    Method,Dataset,Accuracy,Energy,Latency,Sustainability


    """


    methods=data["Method"].unique()



    values=[]



    for method in methods:


        subset=data[

            data["Method"]

            ==

            method

        ]



        values.append(

            subset[metric].mean()

        )



    plt.figure(

        figsize=(8,5)

    )


    plt.bar(

        methods,

        values

    )



    plt.xlabel(

        "Method",

        fontsize=12

    )


    plt.ylabel(

        metric,

        fontsize=12

    )


    plt.title(

        f"{metric} Comparison",

        fontsize=13

    )



    plt.xticks(

        rotation=35,

        ha="right"

    )


    plt.grid(

        axis="y",

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

        metric.lower()+".png"

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
# Dataset-wise Comparison
# =========================================================


def dataset_comparison(

        data,

        metric):


    """

    Plot metric over datasets:


    CIFAR-10

    HAR

    Edge-IIoTset


    """


    datasets=data["Dataset"].unique()



    methods=data["Method"].unique()



    x=np.arange(

        len(datasets)

    )


    width=0.8/(

        len(methods)

    )



    plt.figure(

        figsize=(9,5)

    )



    for i,method in enumerate(methods):


        values=[]


        for dataset in datasets:


            value=data[

                (

                data["Method"]

                ==

                method

                )

                &

                (

                data["Dataset"]

                ==

                dataset

                )

            ][metric].mean()



            values.append(

                value

            )



        plt.bar(

            x+i*width,

            values,

            width,

            label=method

        )




    plt.xticks(

        x+width,

        datasets

    )


    plt.xlabel(

        "Dataset"

    )


    plt.ylabel(

        metric

    )


    plt.title(

        f"Dataset-wise {metric} Evaluation"

    )


    plt.legend(

        fontsize=8

    )


    plt.grid(

        axis="y",

        linestyle="--",

        alpha=0.4

    )


    plt.tight_layout()



    filename=os.path.join(

        OUTPUT_DIR,

        "dataset_"+metric.lower()+".png"

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
# Radar-style Normalized Comparison
# =========================================================


def normalized_metrics_plot(

        data):


    """

    Overall trade-off visualization.


    """


    metrics=METRICS



    methods=data["Method"].unique()



    mean_values=data.groupby(

        "Method"

    )[metrics].mean()



    normalized=(

        mean_values

        -

        mean_values.min()

    )/(

        mean_values.max()

        -

        mean_values.min()

    )



    plt.figure(

        figsize=(8,5)

    )



    for method in methods:


        plt.plot(

            metrics,

            normalized.loc[method],

            marker="o",

            label=method

        )



    plt.ylabel(

        "Normalized Score"

    )


    plt.title(

        "Overall SEA-QAL Performance Trade-off"

    )


    plt.xticks(

        rotation=30

    )


    plt.legend(

        fontsize=8

    )


    plt.grid(

        True,

        linestyle="--",

        alpha=0.4

    )


    plt.tight_layout()



    filename=os.path.join(

        OUTPUT_DIR,

        "overall_tradeoff.png"

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


    data=load_results()



    for metric in METRICS:


        plot_metric(

            data,

            metric

        )


        dataset_comparison(

            data,

            metric

        )



    normalized_metrics_plot(

        data

    )




if __name__=="__main__":


    main()
