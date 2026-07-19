"""
SEA-QAL Semantic Contribution Score Module


Computes the final semantic contribution score
for each candidate update.


Equation:


S_i^t =
wg*g_i^t
+
wu*u_i^t
+
wa*a_i^t


where:

wg + wu + wa = 1


"""


import numpy as np




# =========================================================
# Semantic Score Calculator
# =========================================================


class SemanticScore:



    def __init__(

            self,

            w_g=1/3,

            w_u=1/3,

            w_a=1/3):


        """

        Semantic fusion weights.



        w_g:
            Gradient sensitivity weight


        w_u:
            Prediction uncertainty weight


        w_a:
            Attention concentration weight


        """


        if abs(

            w_g+w_u+w_a-1

        ) > 1e-6:


            raise ValueError(

                "Semantic weights must sum to 1"

            )



        self.w_g=w_g

        self.w_u=w_u

        self.w_a=w_a




    # =====================================================
    # Min-Max Normalization
    # =====================================================


    def normalize(

            self,

            values):


        """

        Normalize semantic indicators

        into [0,1]


        """


        values=np.asarray(

            values,

            dtype=float

        )



        minimum=np.min(values)


        maximum=np.max(values)



        if abs(

            maximum-minimum

        ) < 1e-12:


            return np.zeros_like(

                values

            )



        return (

            values-minimum

        )/(

            maximum-minimum

        )




    # =====================================================
    # Semantic Score Computation
    # =====================================================


    def compute(

            self,

            gradient_scores,

            uncertainty_scores,

            attention_scores):


        """

        Compute semantic contribution score.



        Parameters:

        gradient_scores:
            normalized gradient sensitivity


        uncertainty_scores:
            normalized uncertainty


        attention_scores:
            normalized attention concentration



        Returns:

        semantic contribution S_i


        """



        g=np.asarray(

            gradient_scores

        )


        u=np.asarray(

            uncertainty_scores

        )


        a=np.asarray(

            attention_scores

        )



        # Ensure all values
        # are in [0,1]


        g=self.normalize(g)


        u=self.normalize(u)


        a=self.normalize(a)




        semantic_score=(


            self.w_g*g

            +

            self.w_u*u

            +

            self.w_a*a


        )



        return semantic_score




    # =====================================================
    # Weighted Configuration Update
    # =====================================================


    def update_weights(

            self,

            w_g,

            w_u,

            w_a):


        """

        Used for semantic sensitivity analysis.


        Section 4.4.1



        """


        if abs(

            w_g+w_u+w_a-1

        ) > 1e-6:


            raise ValueError(

                "Weights must sum to 1"

            )


        self.w_g=w_g

        self.w_u=w_u

        self.w_a=w_a




    # =====================================================
    # Explain Contribution
    # =====================================================


    def contribution_analysis(

            self,

            gradient_scores,

            uncertainty_scores,

            attention_scores):


        """

        Returns contribution of
        each semantic component.


        Useful for ablation studies.


        """


        g=np.asarray(

            gradient_scores

        )


        u=np.asarray(

            uncertainty_scores

        )


        a=np.asarray(

            attention_scores

        )



        return {


            "gradient":

            self.w_g*g,


            "uncertainty":

            self.w_u*u,


            "attention":

            self.w_a*a


        }





# =========================================================
# External Interface
# =========================================================


def calculate_semantic_score(

        gradient_scores,

        uncertainty_scores,

        attention_scores,

        w_g=1/3,

        w_u=1/3,

        w_a=1/3):


    """

    Main API used by SEA-QAL framework.


    """


    module=SemanticScore(

        w_g,

        w_u,

        w_a

    )



    return module.compute(

        gradient_scores,

        uncertainty_scores,

        attention_scores

    )





# =========================================================
# Test
# =========================================================


if __name__=="__main__":



    gradient=[

        0.7,

        0.4,

        0.9

    ]


    uncertainty=[

        0.5,

        0.8,

        0.3

    ]


    attention=[

        0.6,

        0.7,

        0.4

    ]



    semantic=(

        calculate_semantic_score(

            gradient,

            uncertainty,

            attention

        )

    )



    print(

        "Semantic Contribution Scores:"

    )


    print(

        semantic

    )
