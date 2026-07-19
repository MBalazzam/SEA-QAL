"""
SEA-QAL Interaction Modeling Module

Computes pairwise interaction terms for QUBO off-diagonal
coefficients.

Equation:

Qij =
lambda_s * R_sem
+
lambda_c * R_com
+
lambda_r * R_res


Interaction components:

1. Semantic redundancy
2. Communication coupling
3. Resource contention


"""


import numpy as np



# =========================================================
# Interaction Model
# =========================================================


class InteractionModel:


    def __init__(

            self,

            lambda_s=1/3,

            lambda_c=1/3,

            lambda_r=1/3):


        """

        QUBO interaction weights.


        lambda_s:
            semantic redundancy


        lambda_c:
            communication coupling


        lambda_r:
            resource contention


        """


        self.lambda_s=lambda_s

        self.lambda_c=lambda_c

        self.lambda_r=lambda_r



    # =====================================================
    # Min-Max Normalization
    # =====================================================


    def normalize(

            self,

            matrix):


        """

        Normalize interaction matrix

        into [0,1]


        """


        matrix=np.asarray(

            matrix,

            dtype=float

        )


        min_value=np.min(matrix)


        max_value=np.max(matrix)



        if abs(max_value-min_value)<1e-12:

            return np.zeros_like(

                matrix

            )



        return (

            matrix-min_value

        )/(

            max_value-min_value

        )




    # =====================================================
    # Semantic Redundancy
    # =====================================================


    def semantic_redundancy(

            self,

            embeddings):


        """

        Equation (15)



        R_sem(i,j)=

        (s_i.s_j)/

        (||s_i|| ||s_j||)



        Cosine similarity


        """


        n=len(

            embeddings

        )


        R=np.zeros(

            (

                n,

                n

            )

        )



        for i in range(n):


            for j in range(n):


                if i==j:

                    continue



                numerator=(

                    np.dot(

                        embeddings[i],

                        embeddings[j]

                    )

                )



                denominator=(

                    np.linalg.norm(

                        embeddings[i]

                    )

                    *

                    np.linalg.norm(

                        embeddings[j]

                    )

                    +

                    1e-8

                )



                R[i,j]=(


                    numerator

                    /

                    denominator

                )



        return self.normalize(

            R

        )




    # =====================================================
    # Communication Coupling
    # =====================================================


    def communication_coupling(

            self,

            data_sizes,

            bandwidth):


        """

        Equation (16)



        R_com(i,j)=

        (d_i*d_j)/(B_t)^2



        """


        n=len(

            data_sizes

        )


        R=np.zeros(

            (

                n,

                n

            )

        )



        for i in range(n):


            for j in range(n):


                if i != j:


                    R[i,j]=(


                        data_sizes[i]

                        *

                        data_sizes[j]

                        /

                        (

                            bandwidth**2

                        )

                    )



        return self.normalize(

            R

        )




    # =====================================================
    # Resource Contention
    # =====================================================


    def resource_contention(

            self,

            resource_requirements,

            max_capacity):


        """

        Equation (17)



        R_res(i,j)=

        (c_i*c_j)/(Cmax)^2



        """


        n=len(

            resource_requirements

        )


        R=np.zeros(

            (

                n,

                n

            )

        )



        for i in range(n):


            for j in range(n):


                if i != j:


                    R[i,j]=(


                        resource_requirements[i]

                        *

                        resource_requirements[j]

                        /

                        (

                            max_capacity**2

                        )

                    )



        return self.normalize(

            R

        )




    # =====================================================
    # Complete Interaction Matrix
    # =====================================================


    def build_interaction_matrix(

            self,

            embeddings,

            data_sizes,

            bandwidth,

            resource_requirements,

            max_capacity):


        """

        Generate complete off-diagonal
        QUBO interaction matrix.


        """


        R_sem=(

            self.semantic_redundancy(

                embeddings

            )

        )



        R_com=(

            self.communication_coupling(

                data_sizes,

                bandwidth

            )

        )



        R_res=(

            self.resource_contention(

                resource_requirements,

                max_capacity

            )

        )



        n=len(

            embeddings

        )


        interaction=np.zeros(

            (

                n,

                n

            )

        )



        for i in range(n):


            for j in range(n):


                if i != j:


                    interaction[i,j]=(


                        self.lambda_s*

                        R_sem[i,j]


                        +

                        self.lambda_c*

                        R_com[i,j]


                        +

                        self.lambda_r*

                        R_res[i,j]

                    )



        return interaction




    # =====================================================
    # Interaction Statistics
    # =====================================================


    def statistics(

            self,

            interaction_matrix):


        """

        Used for analysis and ablation.


        """


        upper=interaction_matrix[

            np.triu_indices(

                interaction_matrix.shape[0],

                k=1

            )

        ]



        return {


            "mean_interaction":

                float(

                    np.mean(

                        upper

                    )

                ),


            "max_interaction":

                float(

                    np.max(

                        upper

                    )

                ),


            "std_interaction":

                float(

                    np.std(

                        upper

                    )

                )

        }




# =========================================================
# Simple Interface
# =========================================================


def compute_interactions(

        embeddings,

        data_sizes,

        bandwidth,

        resource_requirements,

        max_capacity):



    model=InteractionModel()



    return model.build_interaction_matrix(

        embeddings,

        data_sizes,

        bandwidth,

        resource_requirements,

        max_capacity

    )




# =========================================================
# Test
# =========================================================


if __name__=="__main__":


    model=InteractionModel()



    embeddings=[

        np.array([0.2,0.5,0.8]),

        np.array([0.3,0.4,0.7]),

        np.array([0.9,0.1,0.2])

    ]



    interaction=(

        model.build_interaction_matrix(

            embeddings,

            data_sizes=[100,150,120],

            bandwidth=1000,

            resource_requirements=[2,3,1],

            max_capacity=10

        )

    )



    print(

        "Interaction Matrix:"

    )


    print(interaction)



    print(

        model.statistics(

            interaction

        )

    )
