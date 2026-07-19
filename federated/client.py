"""
SEA-QAL Federated Edge Client

Implementation of heterogeneous edge clients
for semantic-energy coupled federated optimization.

Responsibilities:

1. Local training
2. Model update generation
3. Semantic contribution estimation
4. Energy estimation
5. Communication statistics

"""


import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np



# =========================================================
# Federated Edge Client
# =========================================================


class EdgeClient:


    def __init__(

            self,

            client_id,

            model,

            train_loader,

            device="cpu",

            learning_rate=0.001,

            local_epochs=1,

            cpu_frequency=2.0,

            transmission_power=0.5

    ):


        """
        Parameters

        ----------

        client_id:
            unique edge identifier


        model:
            local neural network


        train_loader:
            local non-IID dataset


        device:
            cpu/gpu


        cpu_frequency:
            GHz equivalent computing capability


        transmission_power:
            communication power parameter

        """



        self.client_id = client_id


        self.model = copy.deepcopy(model)


        self.train_loader=train_loader


        self.device=device


        self.local_epochs=local_epochs


        self.optimizer=optim.Adam(

            self.model.parameters(),

            lr=learning_rate

        )


        self.criterion=nn.CrossEntropyLoss()



        # Edge hardware parameters


        self.cpu_frequency=cpu_frequency


        self.transmission_power=transmission_power



        # statistics


        self.training_loss=[]

        self.samples=len(train_loader.dataset)




    # =====================================================
    # Receive Global Model
    # =====================================================


    def update_model(

            self,

            global_weights):


        """

        Synchronize local model

        with global model


        """


        self.model.load_state_dict(

            copy.deepcopy(global_weights)

        )




    # =====================================================
    # Local Training
    # =====================================================


    def train_local_model(self):


        """

        Local federated training.



        Returns

        -------

        updated weights


        """


        self.model.train()



        total_loss=0



        for epoch in range(

                self.local_epochs

        ):


            epoch_loss=0



            for x,y in self.train_loader:


                x=x.to(

                    self.device

                )


                y=y.to(

                    self.device

                )



                self.optimizer.zero_grad()



                output=self.model(x)



                loss=self.criterion(

                    output,

                    y

                )



                loss.backward()



                self.optimizer.step()



                epoch_loss+=loss.item()



            total_loss += epoch_loss



            self.training_loss.append(

                epoch_loss

            )



        return (

            copy.deepcopy(

                self.model.state_dict()

            )

        )




    # =====================================================
    # Compute Local Update Difference
    # =====================================================


    def compute_update(

            self,

            global_weights):


        """

        Calculate local gradient/update.



        Used for:

        semantic contribution estimation


        """


        update={}



        local_weights=self.model.state_dict()



        for key in global_weights:


            update[key]=(

                local_weights[key]

                -

                global_weights[key]

            )



        return update




    # =====================================================
    # Gradient Sensitivity
    # =====================================================


    def gradient_sensitivity(self):


        """

        Approximation of semantic importance.

        Eq. semantic contribution:


        gradient sensitivity signal


        """


        sensitivity=0



        count=0



        for parameter in self.model.parameters():


            if parameter.grad is not None:


                sensitivity += torch.norm(

                    parameter.grad

                ).item()


                count+=1



        if count==0:

            return 0.0



        return sensitivity/count




    # =====================================================
    # Prediction Uncertainty
    # =====================================================


    def prediction_uncertainty(

            self,

            samples=50):


        """

        Estimate uncertainty using entropy.



        """


        self.model.eval()



        entropy_values=[]



        with torch.no_grad():


            for i,(x,_) in enumerate(

                    self.train_loader

            ):


                if i>=samples:

                    break



                x=x.to(

                    self.device

                )


                prediction=self.model(x)



                prob=torch.softmax(

                    prediction,

                    dim=1

                )


                entropy=(

                    -torch.sum(

                        prob*

                        torch.log(

                            prob+1e-8

                        ),

                        dim=1

                    )

                )



                entropy_values.extend(

                    entropy.cpu().numpy()

                )



        return float(

            np.mean(

                entropy_values

            )

        )




    # =====================================================
    # Attention Concentration Approximation
    # =====================================================


    def attention_concentration(self):


        """

        Proxy metric for attention importance.



        Higher value:

        concentrated informative updates



        """


        weights=[]



        for parameter in self.model.parameters():


            weights.append(

                torch.mean(

                    torch.abs(parameter)

                ).item()

            )



        if len(weights)==0:

            return 0



        return float(

            np.std(weights)

        )




    # =====================================================
    # Semantic Contribution Score
    # =====================================================


    def semantic_score(

            self,

            wg=1/3,

            wu=1/3,

            wa=1/3):


        """

        Eq.(8)



        S_i =

        wg*g_i

        +wu*u_i

        +wa*a_i



        """


        g=self.gradient_sensitivity()


        u=self.prediction_uncertainty()


        a=self.attention_concentration()



        score=(

            wg*g

            +

            wu*u

            +

            wa*a

        )



        return score




    # =====================================================
    # Computation Energy
    # =====================================================


    def computation_energy(self):


        """

        Dynamic voltage frequency inspired model.



        """


        cycles=sum(

            [

                p.numel()

                for p in self.model.parameters()

            ]

        )



        energy=(

            cycles*

            self.local_epochs

            /

            (self.cpu_frequency*1e9)

        )



        return energy




    # =====================================================
    # Communication Energy
    # =====================================================


    def communication_energy(

            self):


        """

        Transmission energy



        E=P*t



        """


        model_size=sum(

            [

                p.numel()

                for p in self.model.parameters()

            ]

        )



        transmission_time=(

            model_size/

            1e6

        )



        energy=(

            self.transmission_power

            *

            transmission_time

        )



        return energy




    # =====================================================
    # Latency Estimation
    # =====================================================


    def latency(self):


        computation_time=(

            self.computation_energy()

            /

            self.cpu_frequency

        )


        communication_time=(

            self.communication_energy()

            /

            self.transmission_power

        )


        return (

            computation_time

            +

            communication_time

        )




    # =====================================================
    # Client Report
    # =====================================================


    def get_client_statistics(self):


        """

        Return SEA-QAL metrics

        """


        return {


            "client_id":

                self.client_id,


            "semantic_score":

                self.semantic_score(),


            "energy":

                (

                    self.computation_energy()

                    +

                    self.communication_energy()

                ),


            "latency":

                self.latency(),


            "samples":

                self.samples


        }



# =========================================================
# Testing
# =========================================================


if __name__=="__main__":


    print(

        "SEA-QAL Edge Client Module Loaded"

    )
