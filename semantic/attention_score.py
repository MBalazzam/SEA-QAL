"""
SEA-QAL Attention Concentration Module

Computes attention concentration score
for semantic contribution estimation.


Attention component:

a_i^t


Used in:

S_i^t =
wg*g_i^t +
wu*u_i^t +
wa*a_i^t


"""



import numpy as np
import torch
import torch.nn.functional as F




# =========================================================
# Attention Score Calculator
# =========================================================


class AttentionScore:


    def __init__(

            self,

            temperature=1.0):


        """

        Temperature controls
        attention sharpness.


        Lower temperature:
        sharper attention distribution


        """


        self.temperature=temperature




    # =====================================================
    # Feature Normalization
    # =====================================================


    def normalize_features(

            self,

            features):


        """

        L2 normalization of feature vectors.


        """


        if isinstance(

                features,

                torch.Tensor):


            features=features.detach()



            features=F.normalize(

                features,

                p=2,

                dim=-1

            )



        else:


            norm=np.linalg.norm(

                features,

                axis=-1,

                keepdims=True

            )


            features=(

                features/

                (norm+1e-8)

            )



        return features




    # =====================================================
    # Attention Distribution
    # =====================================================


    def compute_attention(

            self,

            features):


        """

        Generate attention weights
        from feature importance.


        Attention:

        A_i = softmax(||f_i||/T)


        """


        if isinstance(

                features,

                torch.Tensor):


            importance=torch.norm(

                features,

                dim=-1

            )


            attention=F.softmax(

                importance/

                self.temperature,

                dim=0

            )



            return attention.detach().cpu().numpy()



        else:


            importance=np.linalg.norm(

                features,

                axis=-1

            )


            exp_value=np.exp(

                importance/

                self.temperature

            )



            return (

                exp_value/

                np.sum(exp_value)

            )




    # =====================================================
    # Entropy-based Concentration
    # =====================================================


    def entropy_concentration(

            self,

            attention):


        """

        Attention concentration based on entropy.



        High concentration:

        low entropy



        a = 1 - H(A)/log(N)



        """


        attention=np.asarray(

            attention

        )



        n=len(attention)



        entropy=(

            -

            np.sum(

                attention*

                np.log(

                    attention+1e-8

                )

            )

        )



        normalized_entropy=(

            entropy/

            np.log(n+1e-8)

        )



        concentration=(

            1-

            normalized_entropy

        )



        return float(

            np.clip(

                concentration,

                0,

                1

            )

        )




    # =====================================================
    # Maximum Attention Concentration
    # =====================================================


    def max_attention_score(

            self,

            attention):


        """

        Alternative concentration metric.



        Higher maximum attention:

        stronger focus


        """


        return float(

            np.max(

                attention

            )

        )




    # =====================================================
    # Complete Attention Score
    # =====================================================


    def calculate(

            self,

            features):


        """

        Complete attention pipeline.



        Returns:

        attention weights

        concentration score



        """


        features=self.normalize_features(

            features

        )


        attention=(

            self.compute_attention(

                features

            )

        )


        concentration=(

            self.entropy_concentration(

                attention

            )

        )


        return {


            "attention_weights":

                attention,


            "attention_score":

                concentration

        }




# =========================================================
# Batch Processing
# =========================================================


def compute_attention_score(

        feature_batches):


    """

    Compute attention scores
    for multiple clients.


    """


    calculator=AttentionScore()



    scores=[]



    for features in feature_batches:


        result=(

            calculator.calculate(

                features

            )

        )


        scores.append(

            result["attention_score"]

        )



    return scores




# =========================================================
# Test
# =========================================================


if __name__=="__main__":


    calculator=AttentionScore()



    feature=torch.tensor(

        [

            [0.2,0.5,0.8],

            [0.1,0.3,0.4],

            [0.9,0.7,0.6]

        ],

        dtype=torch.float32

    )



    result=(

        calculator.calculate(

            feature

        )

    )



    print(

        "Attention weights:",

        result["attention_weights"]

    )


    print(

        "Attention concentration:",

        result["attention_score"]

    )
