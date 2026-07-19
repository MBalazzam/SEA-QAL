"""
SEA-QAL QUBO Construction Module

Constructs quadratic unconstrained binary optimization
matrix based on semantic-energy coupled optimization.


Diagonal term:

Qii =
alpha*L_i
+
beta*E_i
-
gamma*S_i


Off-diagonal term:

Qij =
lambda_s*R_sem
+
lambda_c*R_com
+
lambda_r*R_res


"""



import numpy as np




# =========================================================
# QUBO Builder
# =========================================================


class QUBOBuilder:


    def __init__(

            self,

            alpha=0.33,

            beta=0.33,

            gamma=0.34,

            lambda_s=1/3,

            lambda_c=1/3,

            lambda_r=1/3):


        """

        Multi-objective weights:

        alpha:
            learning loss


        beta:
            energy consumption


        gamma:
            semantic reward



        Interaction weights:

        lambda_s:
            semantic redundancy


        lambda_c:
            communication coupling


        lambda_r:
            resource contention


        """


        self.alpha=alpha

        self.beta=beta

        self.gamma=gamma



        self.lambda_s=lambda_s

        self.lambda_c=lambda_c

        self.lambda_r=lambda_r




    # =====================================================
    # Normalization
    # =====================================================


    def normalize(

            self,

            values):


        """

        Min-max normalization [0,1]


        """


        values=np.asarray(

            values,

            dtype=float

        )


        minimum=np.min(values)


        maximum=np.max(values)



        if abs(maximum-minimum)<1e-10:

            return np.zeros_like(

                values

            )



        return (

            values-minimum

        )/(

            maximum-minimum

        )




    # =====================================================
    # Semantic Redundancy
    # =====================================================


    def semantic_redundancy(

            self,

            semantic_vectors):


        """

        R_sem equation (15)


        Cosine similarity between
        semantic representations.



        """


        n=len(

            semantic_vectors

        )


        R=np.zeros(

            (

                n,n

            )

        )



        for i in range(n):


            for j in range(n):


                if i==j:

                    continue



                numerator=(

                    np.dot(

                        semantic_vectors[i],

                        semantic_vectors[j]

                    )

                )


                denominator=(

                    np.linalg.norm(

                        semantic_vectors[i]

                    )

                    *

                    np.linalg.norm(

                        semantic_vectors[j]

                    )

                    +

                    1e-8

                )



                R[i,j]=(

                    numerator/

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

        Equation (16):


        Rcom = di*dj / B^2



        """


        n=len(

            data_sizes

        )


        R=np.zeros(

            (

                n,n

            )

        )



        for i in range(n):


            for j in range(n):


                if i!=j:


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

            cpu_requirements,

            max_capacity):


        """

        Equation (17):


        Rres = ci*cj / Cmax^2


        """


        n=len(

            cpu_requirements

        )


        R=np.zeros(

            (

                n,n

            )

        )



        for i in range(n):


            for j in range(n):


                if i!=j:


                    R[i,j]=(

                        cpu_requirements[i]

                        *

                        cpu_requirements[j]

                        /

                        (

                            max_capacity**2

                        )

                    )



        return self.normalize(

            R

        )




    # =====================================================
    # Build QUBO Matrix
    # =====================================================


    def build(

            self,

            loss_values,

            energy_values,

            semantic_scores,

            semantic_vectors,

            data_sizes,

            bandwidth,

            cpu_requirements,

            max_capacity):


        """

        Complete QUBO construction.



        Returns:

            Q matrix



        """



        n=len(

            semantic_scores

        )



        Q=np.zeros(

            (

                n,

                n

            )

        )



        # -------------------------------
        # Diagonal coefficients
        # -------------------------------


        for i in range(n):


            Q[i,i]=(


                self.alpha*

                loss_values[i]


                +

                self.beta*

                energy_values[i]


                -

                self.gamma*

                semantic_scores[i]


            )




        # -------------------------------
        # Interaction coefficients
        # -------------------------------


        R_sem=(

            self.semantic_redundancy(

                semantic_vectors

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

                cpu_requirements,

                max_capacity

            )

        )



        for i in range(n):


            for j in range(i+1,n):


                interaction=(


                    self.lambda_s*

                    R_sem[i,j]


                    +

                    self.lambda_c*

                    R_com[i,j]


                    +

                    self.lambda_r*

                    R_res[i,j]


                )



                Q[i,j]=interaction


                Q[j,i]=interaction




        return Q




# =========================================================
# Simple Interface
# =========================================================


def build_qubo_matrix(

        loss_values,

        energy_values,

        semantic_scores,

        semantic_vectors,

        data_sizes,

        bandwidth,

        cpu_requirements,

        max_capacity):



    builder=QUBOBuilder()



    return builder.build(

        loss_values,

        energy_values,

        semantic_scores,

        semantic_vectors,

        data_sizes,

        bandwidth,

        cpu_requirements,

        max_capacity

    )




# =========================================================
# Test
# =========================================================


if __name__=="__main__":


    builder=QUBOBuilder()



    Q=builder.build(

        loss_values=[0.2,0.3,0.1],

        energy_values=[20,30,15],

        semantic_scores=[0.8,0.6,0.9],

        semantic_vectors=[

            np.array([0.2,0.5]),

            np.array([0.1,0.4]),

            np.array([0.8,0.7])

        ],

        data_sizes=[100,150,120],

        bandwidth=1000,

        cpu_requirements=[2,3,1],

        max_capacity=10

    )



    print(

        "QUBO Matrix:"

    )


    print(Q)
