"""
SEA-QAL CNN Model

Lightweight CNN architecture for
federated edge learning.

Supported datasets:
- CIFAR-10
- Synthetic Edge Dataset

Features:
- Edge-friendly architecture
- PyTorch compatible
- Federated learning ready
- Gradient extraction support

"""


import torch
import torch.nn as nn
import torch.nn.functional as F



# =========================================================
# CNN Feature Extractor
# =========================================================


class CNNFeatureExtractor(nn.Module):


    def __init__(self):

        super(

            CNNFeatureExtractor,

            self

        ).__init__()



        self.features=nn.Sequential(


            # Block 1

            nn.Conv2d(

                3,

                32,

                kernel_size=3,

                padding=1

            ),


            nn.BatchNorm2d(

                32

            ),


            nn.ReLU(),


            nn.MaxPool2d(

                2

            ),




            # Block 2

            nn.Conv2d(

                32,

                64,

                kernel_size=3,

                padding=1

            ),


            nn.BatchNorm2d(

                64

            ),


            nn.ReLU(),


            nn.MaxPool2d(

                2

            ),




            # Block 3

            nn.Conv2d(

                64,

                128,

                kernel_size=3,

                padding=1

            ),


            nn.BatchNorm2d(

                128

            ),


            nn.ReLU(),


            nn.AdaptiveAvgPool2d(

                (1,1)

            )

        )



    def forward(

            self,

            x):


        x=self.features(x)


        x=torch.flatten(

            x,

            1

        )


        return x




# =========================================================
# SEA-QAL CNN Classifier
# =========================================================


class SEAQAL_CNN(nn.Module):


    def __init__(

            self,

            num_classes=10):


        super(

            SEAQAL_CNN,

            self

        ).__init__()



        self.encoder=CNNFeatureExtractor()



        self.classifier=nn.Sequential(


            nn.Linear(

                128,

                64

            ),


            nn.ReLU(),


            nn.Dropout(

                0.3

            ),


            nn.Linear(

                64,

                num_classes

            )

        )




    # -----------------------------------------------------

    # Forward propagation

    # -----------------------------------------------------


    def forward(

            self,

            x):


        feature=self.encoder(

            x

        )


        logits=self.classifier(

            feature

        )


        return logits




    # -----------------------------------------------------

    # Feature extraction

    # -----------------------------------------------------


    def extract_features(

            self,

            x):


        """

        Used for semantic representation.

        """


        return self.encoder(

            x

        )




    # -----------------------------------------------------

    # Prediction uncertainty

    # -----------------------------------------------------


    def prediction_entropy(

            self,

            x):


        """

        Entropy-based uncertainty estimation.


        Used in:

        Eq.(8)

        prediction uncertainty component


        """


        self.eval()



        with torch.no_grad():


            logits=self.forward(

                x

            )


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



        return entropy




    # -----------------------------------------------------

    # Gradient sensitivity

    # -----------------------------------------------------


    def gradient_sensitivity(

            self):


        """

        Gradient-based semantic importance.



        """



        values=[]



        for parameter in self.parameters():


            if parameter.grad is not None:


                values.append(

                    torch.norm(

                        parameter.grad

                    )

                )



        if len(values)==0:


            return torch.tensor(

                0.0

            )



        return torch.mean(

            torch.stack(

                values

            )

        )




# =========================================================
# Model Factory
# =========================================================


def create_cnn_model(

        dataset="CIFAR10"):


    """

    Model initialization interface.

    """


    if dataset.upper()=="CIFAR10":


        classes=10


    else:


        classes=10



    model=SEAQAL_CNN(

        num_classes=classes

    )


    return model




# =========================================================
# Test
# =========================================================


if __name__=="__main__":


    model=create_cnn_model()



    sample=torch.randn(

        4,

        3,

        32,

        32

    )



    output=model(

        sample

    )



    print(

        "Output shape:",

        output.shape

    )



    entropy=model.prediction_entropy(

        sample

    )


    print(

        "Entropy:",

        entropy

    )
