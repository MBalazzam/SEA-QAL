"""
SEA-QAL Computation Energy Model

This module implements computation energy estimation
for heterogeneous edge-cloud computing environments.

The model considers:

- CPU frequency
- computational workload
- CPU cycles
- processor capacitance
- DVFS effect
- execution latency

Used in:
- Energy-aware optimization
- QUBO diagonal coefficient construction
- Edge workload simulation
"""


import numpy as np



# ==========================================================
# Execution Time Model
# ==========================================================


def calculate_execution_time(

        cpu_cycles,

        cpu_frequency):

    """
    Calculate computation latency.

    Formula:

        T = C / f


    Parameters
    ----------
    cpu_cycles :
        Required CPU cycles


    cpu_frequency :
        CPU frequency in GHz


    Returns
    -------
    execution_time seconds

    """


    frequency_hz=(

        cpu_frequency

        *

        1e9

    )


    execution_time=(

        cpu_cycles

        /

        frequency_hz

    )


    return execution_time




# ==========================================================
# DVFS Computation Energy Model
# ==========================================================


def calculate_computation_energy(

        cpu_cycles,

        cpu_frequency,

        capacitance=1e-27):


    """
    Dynamic voltage-frequency scaling energy model.


    Formula:

        E = k * C * f^2 * cycles


    Parameters
    ----------

    cpu_cycles :
        Required processing cycles


    cpu_frequency :
        GHz


    capacitance :
        Effective switched capacitance


    Returns
    -------

    Energy in Joule


    """


    frequency_hz=(

        cpu_frequency

        *

        1e9

    )


    energy=(

        capacitance

        *

        cpu_cycles

        *

        frequency_hz**2

    )


    return energy




# ==========================================================
# CPU Utilization Aware Energy
# ==========================================================


def utilization_energy_model(

        cpu_cycles,

        cpu_frequency,

        utilization,

        idle_power=0.05,

        dynamic_power=0.2):


    """
    More realistic edge CPU energy model.


    Total energy:

        E = (P_idle + P_dynamic*u)*T


    """


    execution_time=(

        calculate_execution_time(

            cpu_cycles,

            cpu_frequency

        )

    )


    power=(

        idle_power

        +

        dynamic_power

        *

        utilization

    )


    energy=(

        power

        *

        execution_time

    )


    return energy




# ==========================================================
# Batch Computation Energy
# ==========================================================


def batch_computation_energy(

        cpu_cycles,

        cpu_frequencies):


    """

    Vectorized energy calculation.


    """


    cpu_cycles=np.asarray(

        cpu_cycles

    )


    cpu_frequencies=np.asarray(

        cpu_frequencies

    )


    frequency_hz=(

        cpu_frequencies

        *

        1e9

    )


    energy=(

        1e-27

        *

        cpu_cycles

        *

        frequency_hz**2

    )


    return energy




# ==========================================================
# Computation Latency
# ==========================================================


def computation_latency(

        cpu_cycles,

        cpu_frequency):


    """

    Compute processing delay.


    """


    return calculate_execution_time(

        cpu_cycles,

        cpu_frequency

    )




# ==========================================================
# Resource Dependency for QUBO
# ==========================================================


def resource_contention(

        computation_i,

        computation_j,

        max_capacity):


    """

    Calculate QUBO resource interaction.


    Corresponds to:


    R_ij^(res)=
    (c_i*c_j)/(C_max)^2


    """


    contention=(

        computation_i

        *

        computation_j

    )/(

        max_capacity**2

    )


    return contention




# ==========================================================
# Energy Normalization
# ==========================================================


def normalize_energy(

        energy_values):


    """

    Min-max normalization

    """


    energy_values=np.asarray(

        energy_values

    )


    e_min=np.min(

        energy_values

    )


    e_max=np.max(

        energy_values

    )


    if e_max-e_min == 0:

        return np.zeros_like(

            energy_values

        )


    normalized=(

        energy_values-e_min

    )/(

        e_max-e_min

    )


    return normalized




# ==========================================================
# Edge Device Energy Profile Generator
# ==========================================================


def generate_edge_energy_profile(

        num_nodes=20,

        min_cpu=1.2,

        max_cpu=2.4):


    """

    Generate heterogeneous edge devices.

    Compatible with Section 4.1.1:

    CPU:
    1.2-2.4 GHz


    """


    cpu_frequency=np.random.uniform(

        min_cpu,

        max_cpu,

        num_nodes

    )


    return cpu_frequency




# ==========================================================
# Test Example
# ==========================================================


if __name__=="__main__":


    nodes=5


    cpu_freq=generate_edge_energy_profile(

        nodes

    )


    workload=np.random.uniform(

        1e8,

        5e8,

        nodes

    )


    energy=batch_computation_energy(

        workload,

        cpu_freq

    )


    latency=calculation_latency = [

        computation_latency(

            workload[i],

            cpu_freq[i]

        )

        for i in range(nodes)

    ]


    print(
        "CPU Frequencies (GHz):"
    )

    print(cpu_freq)


    print(
        "\nComputation Energy (J):"
    )

    print(energy)


    print(
        "\nLatency (s):"
    )

    print(latency)
