"""
SEA-QAL Communication Energy Model

This module implements the communication energy estimation
for cloud-edge/federated learning scenarios.

The model considers:

- transmitted data size
- bandwidth
- transmission power
- channel quality
- communication distance
- transmission time

Used in:
- Energy-aware optimization
- QUBO construction
- Synthetic Edge Dataset generation
"""


import numpy as np



# ==========================================================
# Basic Communication Energy Model
# ==========================================================


def calculate_transmission_time(

        data_size_MB,

        bandwidth_Mbps):

    """
    Transmission latency model.

    Parameters
    ----------
    data_size_MB :
        Data payload size in MB

    bandwidth_Mbps :
        Available bandwidth Mbps


    Returns
    -------
    time_seconds
    """


    data_bits = (

        data_size_MB

        *

        8

        *

        1024**2

    )


    bandwidth_bits=(

        bandwidth_Mbps

        *

        10**6

    )


    transmission_time=(

        data_bits /

        bandwidth_bits

    )


    return transmission_time




# ==========================================================
# Communication Energy
# ==========================================================


def calculate_communication_energy(

        data_size_MB,

        bandwidth_Mbps,

        transmission_power_W=0.5):


    """
    Calculate communication energy.

    Formula:

        E_comm = P_tx * T_tx


    Parameters
    ----------

    data_size_MB :
        transmitted data size


    bandwidth_Mbps :
        communication bandwidth


    transmission_power_W :
        wireless transmission power


    Returns
    -------

    energy_Joule

    """


    tx_time = calculate_transmission_time(

        data_size_MB,

        bandwidth_Mbps

    )


    energy=(

        transmission_power_W

        *

        tx_time

    )


    return energy




# ==========================================================
# Channel-aware Communication Energy
# ==========================================================


def channel_aware_energy(

        data_size_MB,

        bandwidth_Mbps,

        distance_meter,

        path_loss_exponent=3.5,

        reference_power=0.1):


    """
    Wireless channel-aware energy model.


    Energy increases with:

    - distance

    - path loss


    """


    path_loss=(

        distance_meter

        **

        path_loss_exponent

    )



    effective_power=(

        reference_power

        *

        path_loss

    )



    tx_time=calculate_transmission_time(

        data_size_MB,

        bandwidth_Mbps

    )



    energy=(

        effective_power

        *

        tx_time

    )


    return energy




# ==========================================================
# Batch Communication Energy
# ==========================================================


def batch_communication_energy(

        data_sizes,

        bandwidths,

        transmission_power=0.5):


    """

    Vectorized communication energy calculation.


    Parameters

    ----------

    data_sizes :
        array of MB values


    bandwidths :
        array of Mbps values


    Returns

    -------

    energies

    """


    data_sizes=np.asarray(

        data_sizes

    )


    bandwidths=np.asarray(

        bandwidths

    )


    times=(

        data_sizes

        *

        8

        *

        1024**2

    )/(

        bandwidths

        *

        10**6

    )


    energies=(

        transmission_power

        *

        times

    )


    return energies




# ==========================================================
# Communication Coupling for QUBO
# ==========================================================


def communication_coupling(

        data_i,

        data_j,

        bandwidth):


    """

    Compute pairwise communication coupling term.

    Corresponds to:

    R_ij^(com)=d_i*d_j/B^2


    Used in QUBO off-diagonal coefficient.


    """


    coupling=(

        data_i

        *

        data_j

    )/(

        bandwidth**2

    )


    return coupling




# ==========================================================
# Normalization
# ==========================================================


def normalize_energy(

        energy_values):


    """

    Min-max normalization

    """



    energy_values=np.asarray(

        energy_values

    )


    minimum=np.min(

        energy_values

    )


    maximum=np.max(

        energy_values

    )



    if maximum-minimum==0:

        return np.zeros_like(

            energy_values

        )


    normalized=(

        energy_values-minimum

    )/(

        maximum-minimum

    )


    return normalized




# ==========================================================
# Example Test
# ==========================================================


if __name__=="__main__":


    data=np.array(

        [

            2,

            5,

            10

        ]

    )


    bandwidth=np.array(

        [

            20,

            15,

            10

        ]

    )


    energy=batch_communication_energy(

        data,

        bandwidth

    )


    print(

        "Communication Energy (J):"

    )

    print(

        energy

    )



    coupling=communication_coupling(

        data_i=5,

        data_j=8,

        bandwidth=20

    )


    print(

        "QUBO Communication Coupling:"

    )


    print(

        coupling

    )
