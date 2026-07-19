"""
SEA-QAL Prediction Uncertainty Module


Computes predictive uncertainty of local updates
using entropy-based uncertainty estimation.


Prediction uncertainty:


u_i^t =
- sum_c p_c log(p_c)


where:

p_c = predicted probability of class c



The normalized uncertainty is used in:


S_i^t =
wg*g_i^t
+
wu*u_i^t
+
wa*a_i^t


"""


import numpy as np
import torch
import torch.nn.functional as F




# =========================================================
# Uncertainty Calculator
# =========================================================


class PredictionUncertainty:



    def __init__(

            self,

            epsilon=1e-8):


        self.epsilon=epsilon


        self.history=[]




    # =====================================================
    # Entropy Calculation
    # =====================================================


    def entropy(

            self,

            probabilities):


        """

        Predictive entropy:


        H(p)= -sum(p log p)



        """


        probabilities=np.asarray(

            probabilities,

            dtype=float

        )


        probabilities=np.clip(

            probabilities,

            self.epsilon,

            1.0

        )



        uncertainty=(

            -

            np.sum(

                probabilities

                *

                np.log(probabilities)

            )

        )



        return float(

            uncertainty

        )




    # =====================================================
    # Batch Entropy
    # =====================================================


    def batch_entropy(

            self,

            predictions):


        """

        Compute uncertainty
        for multiple samples.


        predictions:

        shape:

        [samples, classes]



        """


        scores=[]



        for prediction in predictions:


            scores.append(

                self.entropy(

                    prediction

                )

            )



        return np.array(

            scores

        )




    # =====================================================
    # Neural Network Prediction
    # =====================================================


    def compute_from_model(

            self,

            model,

            data,

            device="cpu"):


        """

        Compute uncertainty directly
        from neural network output.



        """


        model.eval()



        with torch.no_grad():


            data=data.to(

                device

            )


            logits=model(

                data

            )


            probabilities=F.softmax(

                logits,

                dim=1

            )


        probabilities=(

            probabilities

            .cpu()

            .numpy()

        )


        uncertainty=(

            self.batch_entropy(

                probabilities

            )

        )



        self.history.extend(

            uncertainty.tolist()

        )



        return uncertainty




    # =====================================================
    # Monte Carlo Dropout Uncertainty
    # =====================================================


    def mc_dropout_uncertainty(

            self,

            model,

            data,

            samples=10,

            device="cpu"):


        """

        Bayesian approximation using
        Monte Carlo dropout.


        """


        predictions=[]



        model.train()



        with torch.no_grad():


            for _ in range(samples):


                output=model(

                    data.to(device)

                )


                probability=(

                    F.softmax(

                        output,

                        dim=1

                    )

                    .cpu()

                    .numpy()

                )


                predictions.append(

                    probability

                )



        predictions=np.array(

            predictions

        )



        mean_prediction=np.mean(

            predictions,

            axis=0

        )



        uncertainty=(

            self.entropy(

                mean_prediction

            )

        )



        return uncertainty




    # =====================================================
    # Normalization
    # =====================================================


    def normalize(

            self,

            values):


        """

        Min-Max normalization:


        [0,1]


        """


        values=np.asarray(

            values,

            dtype=float

        )



        min_value=np.min(

            values

        )


        max_value=np.max(

            values

        )



        if abs(

            max_value-min_value

        ) < 1e-12:


            return np.zeros_like(

                values

            )



        normalized=(

            values-min_value

        )/(

            max_value-min_value

        )



        return normalized




    # =====================================================
    # Complete Pipeline
    # =====================================================


    def calculate(

            self,

            probabilities):


        """

        Complete uncertainty pipeline.


        Input:

        probability vectors


        Output:

        normalized uncertainty scores


        """


        raw_uncertainty=(

            self.batch_entropy(

                probabilities

            )

        )


        normalized=(

            self.normalize(

                raw_uncertainty

            )

        )


        return normalized




    # =====================================================
    # Statistics
    # =====================================================


    def statistics(

            self,

            values):


        values=np.asarray(

            values

        )


        return {


            "mean":

            float(

                np.mean(values)

            ),


            "std":

            float(

                np.std(values)

            ),


            "max":

            float(

                np.max(values)

            ),


            "min":

            float(

                np.min(values)

            )

        }





# =========================================================
# External Interface
# =========================================================


def compute_uncertainty(

        probability_vectors):


    """

    API used by semantic_encoder.py


    """


    module=PredictionUncertainty()



    return module.calculate(

        probability_vectors

    )




# =========================================================
# Test
# =========================================================


if __name__=="__main__":


    predictions=np.array(

        [

            [0.9,0.05,0.05],

            [0.4,0.35,0.25],

            [0.6,0.3,0.1]

        ]

    )



    uncertainty_module=(

        PredictionUncertainty()

    )



    scores=(

        uncertainty_module.calculate(

            predictions

        )

    )



    print(

        "Normalized uncertainty scores:"

    )


    print(scores)
