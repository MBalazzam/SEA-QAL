"""
SEA-QAL Convergence Visualization


Generate convergence curves for:

- SEA-QAL
- QHRMOF
- Quantum-PSO Offloading
- Boltzmann-Bayesian
- AI Predictive Scaling


Figure corresponds to:

Section 4.7
Convergence and Computational Control


"""



import os
import numpy as np
import matplotlib.pyplot as plt




# =========================================================
# Configuration
# =========================================================


RESULT_PATH = "../results/convergence"


OUTPUT_PATH = "../figures/convergence_curve.png"



METHODS = {


    "SEA-QAL":

    "seaqal.npy",


    "QHRMOF":

    "qhrmof.npy",


    "Quantum-PSO":

    "quantum_pso.npy",


    "Boltzmann-Bayesian":

    "boltzmann.npy",


    "AI Predictive Scaling":

    "ai_scaling.npy"

}




# =========================================================
# Load convergence history
# =========================================================


def load_history(file):


    path=os.path.join(

        RESULT_PATH,

        file

    )


    if not os.path.exists(path):


        raise FileNotFoundError(

            f"Missing file: {path}"

        )



    data=np.load(

        path

    )



    return data




# =========================================================
# Normalize Objective
# =========================================================


def normalize_curve(values):


    """

    Normalize objective:

    F_norm=(F-Fmin)/(Fmax-Fmin)


    """


    values=np.asarray(

        values

    )



    f_min=np.min(values)


    f_max=np.max(values)



    if abs(

        f_max-f_min

    ) < 1e-12:


        return np.zeros_like(

            values

        )



    return (

        values-f_min

    )/(

        f_max-f_min

    )




# =========================================================
# Mean and STD over trials
# =========================================================


def process_trials(data):


    """

    Input:

    [trials, epochs]



    Output:

    mean curve

    std curve


    """


    if data.ndim==1:


        data=data.reshape(

            1,

            -1

        )



    normalized=[]



    for trial in data:


        normalized.append(

            normalize_curve(

                trial

            )

        )



    normalized=np.array(

        normalized

    )



    mean=np.mean(

        normalized,

        axis=0

    )


    std=np.std(

        normalized,

        axis=0

    )


    return mean,std




# =========================================================
# Main Plot
# =========================================================


def plot_convergence():



    plt.figure(

        figsize=(8,5)

    )



    for method,file in METHODS.items():


        data=load_history(

            file

        )


        mean,std=(

            process_trials(

                data

            )

        )



        epochs=np.arange(

            1,

            len(mean)+1

        )



        plt.plot(

            epochs,

            mean,

            linewidth=2,

            label=method

        )



        plt.fill_between(

            epochs,

            mean-std,

            mean+std,

            alpha=0.15

        )



    plt.xlabel(

        "Optimization Iterations",

        fontsize=12

    )


    plt.ylabel(

        "Normalized Objective Value",

        fontsize=12

    )



    plt.title(

        "Convergence Characteristics of SEA-QAL and Baselines",

        fontsize=13

    )



    plt.grid(

        True,

        linestyle="--",

        alpha=0.4

    )



    plt.legend(

        fontsize=9

    )


    plt.tight_layout()



    os.makedirs(

        os.path.dirname(

            OUTPUT_PATH

        ),

        exist_ok=True

    )


    plt.savefig(

        OUTPUT_PATH,

        dpi=600,

        bbox_inches="tight"

    )



    plt.close()



    print(

        "Saved:",

        OUTPUT_PATH

    )





# =========================================================
# Convergence Epoch Calculation
# =========================================================


def calculate_convergence_epoch(

        objective,

        epsilon=1e-4,

        patience=10):


    """

    Implements paper stopping criterion:



    DeltaF < epsilon

    for consecutive patience iterations.



    """


    objective=np.asarray(

        objective

    )



    counter=0



    for k in range(

        1,

        len(objective)

    ):


        delta=(

            abs(

                objective[k]

                -

                objective[k-1]

            )

            /

            (

                abs(

                    objective[k-1]

                )

                +

                1e-12

            )

        )



        if delta < epsilon:


            counter +=1


        else:


            counter=0



        if counter >= patience:


            return k+1



    return len(objective)





# =========================================================
# Test
# =========================================================


if __name__=="__main__":


    plot_convergence()



    print(

        "Convergence plotting completed."

    )
