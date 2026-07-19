"""
SEA-QAL Gradient Sensitivity Module


Computes gradient-based semantic importance
of local client updates.


Gradient sensitivity:

g_i^t = || grad_theta L_i^t ||_2


Normalized:

g_tilde =
(g-gmin)/(gmax-gmin)


Used in:

S_i^t =
wg*g_i +
wu*u_i +
wa*a_i


"""


import numpy as np
import torch




# =========================================================
# Gradient Sensitivity Calculator
# =========================================================


class GradientSensitivity:



    def __init__(

            self):


        self.history=[]




    # =====================================================
    # Gradient Norm
    # =====================================================


    def compute_gradient_norm(

            self,

            model):


        """

        Compute:

        ||grad L||2


        from model parameters.


        """


        total_norm=0.0



        for parameter in model.parameters():


            if parameter.grad is not None:


                gradient=parameter.grad.data



                param_norm=torch.norm(

                    gradient,

                    p=2

                )



                total_norm += (

                    param_norm.item()**2

                )



        return float(

            np.sqrt(

                total_norm

            )

        )




    # =====================================================
    # Loss-based Gradient Computation
    # =====================================================


    def compute_from_batch(

            self,

            model,

            data,

            target,

            criterion,

            device="cpu"):


        """

        Compute gradient sensitivity
        for one local update.


        """


        model.zero_grad()



        data=data.to(

            device

        )


        target=target.to(

            device

        )



        output=model(

            data

        )



        loss=criterion(

            output,

            target

        )



        loss.backward()



        gradient_norm=(

            self.compute_gradient_norm(

                model

            )

        )



        self.history.append(

            gradient_norm

        )



        return gradient_norm




    # =====================================================
    # Batch Client Evaluation
    # =====================================================


    def compute_client_sensitivity(

            self,

            clients):


        """

        Calculate gradient sensitivity
        for multiple edge clients.


        Each client should provide:

        get_gradient_norm()



        """


        scores=[]



        for client in clients:


            score=(

                client.get_gradient_norm()

            )


            scores.append(

                score

            )



        return scores




    # =====================================================
    # Min-Max Normalization
    # =====================================================


    def normalize(

            self,

            values):


        """

        Normalize gradient sensitivity
        into [0,1]


        """


        values=np.asarray(

            values,

            dtype=float

        )


        minimum=np.min(

            values

        )


        maximum=np.max(

            values

        )



        if abs(

            maximum-minimum

        ) < 1e-12:


            return np.zeros_like(

                values

            )



        normalized=(

            values-minimum

        )/(

            maximum-minimum

        )



        return normalized




    # =====================================================
    # Complete Pipeline
    # =====================================================


    def calculate(

            self,

            raw_gradients):


        """

        Complete gradient sensitivity pipeline.



        Input:

            raw gradient values


        Output:

            normalized gradient scores



        """


        normalized=(

            self.normalize(

                raw_gradients

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
# Simple Interface
# =========================================================


def compute_gradient_sensitivity(

        gradient_values):


    """

    External API used by semantic_encoder.py



    """


    module=GradientSensitivity()



    return module.calculate(

        gradient_values

    )





# =========================================================
# Test Example
# =========================================================


if __name__=="__main__":


    gradients=[

        0.25,

        0.80,

        0.40,

        1.20,

        0.60

    ]



    module=GradientSensitivity()



    normalized=(

        module.calculate(

            gradients

        )

    )



    print(

        "Raw gradients:",

        gradients

    )


    print(

        "Normalized gradient sensitivity:",

        normalized

    )
