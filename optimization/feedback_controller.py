"""
SEA-QAL Adaptive Feedback Controller

This module implements the closed-loop feedback
mechanism for dynamic adjustment of multi-objective
optimization weights.

Controls:

alpha_t:
Learning loss importance

beta_t:
Energy consumption importance

gamma_t:
Semantic reward importance


Based on:

Objective function:

alpha_t L_i^t
+
beta_t E_i^t
-
gamma_t S_i^t


"""


import numpy as np




# =========================================================
# Feedback Controller
# =========================================================


class FeedbackController:


    def __init__(

            self,

            alpha=0.33,

            beta=0.33,

            gamma=0.34,

            energy_budget=100,

            learning_rate=0.05,

            min_weight=0.1,

            max_weight=0.8):


        """

        Parameters

        ----------


        alpha:
            learning importance


        beta:
            energy importance


        gamma:
            semantic importance



        """


        self.alpha=alpha

        self.beta=beta

        self.gamma=gamma



        self.energy_budget=energy_budget


        self.learning_rate=learning_rate


        self.min_weight=min_weight


        self.max_weight=max_weight



        self.history=[]




    # =====================================================
    # Weight Normalization
    # =====================================================


    def normalize_weights(self):


        """

        Ensure:

        alpha+beta+gamma=1


        """


        total=(

            self.alpha

            +

            self.beta

            +

            self.gamma

        )


        self.alpha/=total

        self.beta/=total

        self.gamma/=total




    # =====================================================
    # Energy Feedback
    # =====================================================


    def update_energy_feedback(

            self,

            current_energy):


        """

        Increase energy importance
        when budget violation occurs.



        """


        deviation=(

            current_energy

            -

            self.energy_budget

        )/self.energy_budget



        if deviation > 0:


            # energy shortage

            self.beta += (

                self.learning_rate

                *

                deviation

            )


            self.alpha -= (

                self.learning_rate

                *

                deviation

                *

                0.5

            )


            self.gamma -= (

                self.learning_rate

                *

                deviation

                *

                0.5

            )


        else:


            # sufficient energy


            self.beta -= (

                self.learning_rate

                *

                abs(deviation)

            )


            self.gamma += (

                self.learning_rate

                *

                abs(deviation)

                *

                0.5

            )



        return self.get_weights()




    # =====================================================
    # Accuracy Feedback
    # =====================================================


    def update_accuracy_feedback(

            self,

            current_accuracy,

            target_accuracy):


        """

        Increase learning importance
        if accuracy falls below target.



        """


        error=(

            target_accuracy

            -

            current_accuracy

        )



        if error > 0:


            self.alpha += (

                self.learning_rate

                *

                error

                /

                100

            )


            self.gamma += (

                self.learning_rate

                *

                error

                /

                200

            )


            self.beta -= (

                self.learning_rate

                *

                error

                /

                300

            )


        return self.get_weights()




    # =====================================================
    # Latency Feedback
    # =====================================================


    def update_latency_feedback(

            self,

            current_latency,

            latency_threshold):


        """

        Increase energy/resource awareness
        under latency violation.



        """


        if current_latency > latency_threshold:


            self.beta += (

                self.learning_rate*0.5

            )


            self.alpha -= (

                self.learning_rate*0.25

            )


            self.gamma -= (

                self.learning_rate*0.25

            )


        return self.get_weights()




    # =====================================================
    # Semantic Feedback
    # =====================================================


    def update_semantic_feedback(

            self,

            semantic_quality):


        """

        Increase semantic importance
        when semantic quality is low.



        """


        if semantic_quality < 0.5:


            self.gamma += (

                self.learning_rate

                *

                (0.5-semantic_quality)

            )


            self.alpha -= (

                self.learning_rate

                *

                0.5

                *

                (0.5-semantic_quality)

            )



        return self.get_weights()




    # =====================================================
    # Complete Closed-loop Update
    # =====================================================


    def update(

            self,

            current_energy,

            current_accuracy,

            current_latency,

            semantic_quality,

            target_accuracy,

            latency_threshold):


        """

        Complete feedback loop.



        """


        self.update_energy_feedback(

            current_energy

        )


        self.update_accuracy_feedback(

            current_accuracy,

            target_accuracy

        )


        self.update_latency_feedback(

            current_latency,

            latency_threshold

        )


        self.update_semantic_feedback(

            semantic_quality

        )



        self.clip_weights()


        self.normalize_weights()



        weights=self.get_weights()



        self.history.append(

            weights.copy()

        )



        return weights




    # =====================================================
    # Weight Constraints
    # =====================================================


    def clip_weights(self):


        self.alpha=np.clip(

            self.alpha,

            self.min_weight,

            self.max_weight

        )


        self.beta=np.clip(

            self.beta,

            self.min_weight,

            self.max_weight

        )


        self.gamma=np.clip(

            self.gamma,

            self.min_weight,

            self.max_weight

        )




    # =====================================================
    # Get Current Weights
    # =====================================================


    def get_weights(self):


        return {


            "alpha":

            round(

                self.alpha,

                4

            ),


            "beta":

            round(

                self.beta,

                4

            ),


            "gamma":

            round(

                self.gamma,

                4

            )

        }




    # =====================================================
    # Reset Controller
    # =====================================================


    def reset(self):


        self.alpha=0.33

        self.beta=0.33

        self.gamma=0.34


        self.history=[]




# =========================================================
# Test
# =========================================================


if __name__=="__main__":


    controller=FeedbackController(

        energy_budget=100

    )


    print(

        "Initial:",

        controller.get_weights()

    )



    new_weights=controller.update(

        current_energy=130,

        current_accuracy=91,

        current_latency=45,

        semantic_quality=0.4,

        target_accuracy=93,

        latency_threshold=40

    )



    print(

        "Updated:",

        new_weights

    )
