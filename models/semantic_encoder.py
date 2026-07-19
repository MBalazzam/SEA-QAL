"""
SEA-QAL Semantic Encoder Module

This module implements semantic contribution estimation
for federated edge updates.

Components:

1. Gradient sensitivity
2. Prediction uncertainty
3. Attention concentration
4. Semantic fusion score


Based on:

S_i^t =
w_g*g_i^t +
w_u*u_i^t +
w_a*a_i^t


"""


import torch
import torch.nn.functional as F

import numpy as np




# =========================================================
# Semantic Encoder
# =========================================================


class SemanticEncoder:


    def __init__(

            self,

            wg=1/3,

            wu=1/3,

            wa=1/3):


        """
        Semantic fusion weights.

        Eq. (8)


        """


        self.wg=wg

        self.wu=wu

        self.wa=wa



    # =====================================================
    # Min-Max Normalization
    # =====================================================


    def normalize(

            self,

            values):


        """

        Normalize heterogeneous semantic signals
        into [0,1].


        """


        values=np.asarray(

            values,

            dtype=float

        )


        minimum=np.min(values)


        maximum=np.max(values)



        if maximum-minimum < 1e-8:


            return np.ones_like(

                values

            )



        return (

            values-minimum

        )/(

            maximum-minimum

        )



    # =====================================================
    # Gradient Sensitivity
    # =====================================================


    def gradient_sensitivity(

            self,

            model):


        """

        Gradient importance estimation.



        Represents:

        g_i^t


        """


        gradients=[]



        for parameter in model.parameters():


            if parameter.grad is not None:


                gradients.append(

                    torch.norm(

                        parameter.grad

                    ).item()

                )



        if len(gradients)==0:


            return 0.0



        return np.mean(

            gradients

        )




    # =====================================================
    # Prediction Uncertainty
    # =====================================================


    def prediction_uncertainty(

            self,

            logits):


        """

        Entropy-based uncertainty.



        u_i^t


        """


        probabilities=torch.softmax(

            logits,

            dim=1

        )


        entropy=(

            -torch.sum(

                probabilities*

                torch.log(

                    probabilities+1e-8

                ),

                dim=1

            )

        )



        return torch.mean(

            entropy

        ).item()




    # =====================================================
    # Attention Concentration
    # =====================================================


    def attention_concentration(

            self,

            feature_vectors):


        """

        Attention concentration estimation.



        a_i^t



        Higher variance:

        more concentrated semantic features


        """


        if isinstance(

                feature_vectors,

                torch.Tensor):


            feature_vectors=(

                feature_vectors.detach()

                .cpu()

                .numpy()

            )



        attention=np.abs(

            feature_vectors

        )



        attention=attention/(

            np.sum(

                attention,

                axis=-1,

                keepdims=True

            )

            +

            1e-8

        )



        concentration=np.mean(

            np.max(

                attention,

                axis=-1

            )

        )



        return concentration




    # =====================================================
    # Semantic Contribution Score
    # =====================================================


    def semantic_score(

            self,

            gradient,

            uncertainty,

            attention):


        """

        Equation (8):



        S_i^t =
        wg*g_i^t
        +
        wu*u_i^t
        +
        wa*a_i^t



        """


        components=np.array(

            [

                gradient,

                uncertainty,

                attention

            ]

        )



        normalized=self.normalize(

            components

        )



        g,u,a=normalized



        score=(

            self.wg*g

            +

            self.wu*u

            +

            self.wa*a

        )



        return float(

            score

        )




    # =====================================================
    # Encode Client Update
    # =====================================================


    def encode(

            self,

            model,

            logits,

            features):


        """

        Complete semantic encoding pipeline.



        Returns:

        g,u,a,S



        """


        gradient=(

            self.gradient_sensitivity(

                model

            )

        )


        uncertainty=(

            self.prediction_uncertainty(

                logits

            )

        )


        attention=(

            self.attention_concentration(

                features

            )

        )



        score=(

            self.semantic_score(

                gradient,

                uncertainty,

                attention

            )

        )



        return {


            "gradient_sensitivity":

                gradient,


            "uncertainty":

                uncertainty,


            "attention":

                attention,


            "semantic_score":

                score

        }




# =========================================================
# Batch Semantic Encoding
# =========================================================


def encode_clients(

        clients_data):


    """

    Encode multiple edge clients.



    clients_data:

    [

      {

       model,

       logits,

       features

      }

    ]



    """


    encoder=SemanticEncoder()



    results=[]



    for item in clients_data:


        result=encoder.encode(

            item["model"],

            item["logits"],

            item["features"]

        )


        results.append(

            result

        )



    return results




# =========================================================
# Test
# =========================================================


if __name__=="__main__":


    encoder=SemanticEncoder()



    example=encoder.semantic_score(

        gradient=0.8,

        uncertainty=0.4,

        attention=0.7

    )



    print(

        "Semantic Contribution:",

        example

    )
