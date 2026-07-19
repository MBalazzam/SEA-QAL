"""
SEA-QAL Federated Model Manager

Manages:

1. Global model initialization
2. Local model distribution
3. Model synchronization
4. Federated aggregation
5. Model update handling


Compatible with:
- cnn.py
- client.py
- aggregation.py
- train.py


"""



import copy
import torch

from collections import OrderedDict




# =========================================================
# Federated Model Manager
# =========================================================


class FederatedModel:


    def __init__(

            self,

            base_model,

            device="cpu"):


        """

        Parameters
        ----------

        base_model:
            CNN model architecture


        device:
            cpu / cuda


        """


        self.device=device



        # Global model

        self.global_model=(

            copy.deepcopy(

                base_model

            )

            .to(device)

        )



        self.round=0




    # =====================================================
    # Get Global Weights
    # =====================================================


    def get_global_weights(self):


        """

        Return global parameters.


        Used for:

        broadcasting to edge clients


        """


        return copy.deepcopy(

            self.global_model.state_dict()

        )




    # =====================================================
    # Initialize Local Models
    # =====================================================


    def create_local_models(

            self,

            num_clients):


        """

        Create identical local models
        for edge nodes.


        """


        models=[]



        for _ in range(num_clients):


            local_model=(

                copy.deepcopy(

                    self.global_model

                )

            )


            models.append(

                local_model

            )


        return models




    # =====================================================
    # Update Global Model
    # =====================================================


    def update_global_model(

            self,

            aggregated_weights):


        """

        Replace global model parameters.

        """


        self.global_model.load_state_dict(

            aggregated_weights

        )


        self.round+=1




    # =====================================================
    # FedAvg Aggregation
    # =====================================================


    def fedavg(

            self,

            client_weights):


        """

        Standard FedAvg:



        W = Σ(n_i/n) W_i



        """


        aggregated=OrderedDict()



        for key in client_weights[0]:


            aggregated[key]=torch.mean(

                torch.stack(

                    [

                        w[key].float()

                        for w in client_weights

                    ]

                ),

                dim=0

            )



        return aggregated




    # =====================================================
    # Semantic Energy Aggregation
    # =====================================================


    def seaqal_aggregation(

            self,

            client_weights,

            semantic_scores,

            energy_values,

            latency_values,

            alpha=0.33,

            beta=0.33,

            gamma=0.34):


        """

        SEA-QAL adaptive aggregation.



        Contribution:



        C_i =

        alpha*S_i

        + beta*(1/E_i)

        + gamma*(1/L_i)



        """



        scores=[]



        for i in range(

                len(client_weights)

        ):



            energy_eff=(

                1/

                (

                    energy_values[i]+1e-8

                )

            )



            latency_eff=(

                1/

                (

                    latency_values[i]+1e-8

                )

            )



            score=(

                alpha*

                semantic_scores[i]

                +

                beta*

                energy_eff

                +

                gamma*

                latency_eff

            )



            scores.append(

                score

            )



        # Normalize


        scores=torch.tensor(

            scores,

            dtype=torch.float32

        )



        scores=(

            scores/

            torch.sum(scores)

        )



        aggregated=OrderedDict()



        for key in client_weights[0]:


            aggregated[key]=sum(

                [

                    client_weights[i][key]

                    *

                    scores[i]

                    for i in range(

                        len(client_weights)

                    )

                ]

            )



        return aggregated,scores




    # =====================================================
    # Apply Client Updates
    # =====================================================


    def apply_updates(

            self,

            updates):


        """

        Convert local updates into
        aggregated global parameters.



        """


        aggregated=self.fedavg(

            updates

        )


        self.update_global_model(

            aggregated

        )



        return self.get_global_weights()




    # =====================================================
    # Model Evaluation
    # =====================================================


    def evaluate(

            self,

            data_loader,

            criterion):


        """

        Evaluate global model.



        Returns:

        accuracy

        loss


        """


        self.global_model.eval()



        correct=0

        total=0

        loss_sum=0



        with torch.no_grad():


            for x,y in data_loader:


                x=x.to(

                    self.device

                )


                y=y.to(

                    self.device

                )



                output=self.global_model(

                    x

                )



                loss=criterion(

                    output,

                    y

                )



                loss_sum+=loss.item()



                prediction=(

                    torch.argmax(

                        output,

                        dim=1

                    )

                )



                correct += (

                    prediction==y

                ).sum().item()



                total+=y.size(0)



        accuracy=(

            100*

            correct/

            total

        )


        return {

            "accuracy":

            accuracy,


            "loss":

            loss_sum/

            len(data_loader)

        }




    # =====================================================
    # Save Model
    # =====================================================


    def save_checkpoint(

            self,

            path="global_model.pt"):


        torch.save(

            {

            "round":

            self.round,


            "model_state":

            self.global_model.state_dict()

            },

            path

        )




    # =====================================================
    # Load Model
    # =====================================================


    def load_checkpoint(

            self,

            path):


        checkpoint=torch.load(

            path,

            map_location=self.device

        )


        self.global_model.load_state_dict(

            checkpoint["model_state"]

        )


        self.round=(

            checkpoint["round"]

        )




# =========================================================
# Testing
# =========================================================


if __name__=="__main__":


    from cnn import create_cnn_model



    model=create_cnn_model()



    fed_model=FederatedModel(

        model

    )



    local_models=(

        fed_model.create_local_models(

            5

        )

    )


    print(

        "Number of local models:",

        len(local_models)

    )


    print(

        "Federated model initialized."

    )
