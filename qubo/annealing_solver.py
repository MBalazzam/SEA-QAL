"""
SEA-QAL Quantum-Inspired Annealing Solver

Classical optimizer inspired by quantum annealing
principles for solving QUBO problems.

Objective:

H(x)=x^T Q x


Transition:

P(xa -> xb)=
min(1, exp(-deltaH/Tk)*psi_k)


where:

psi_k = exp(-eta*k)


The solver runs completely on classical hardware.


"""


import numpy as np



# =========================================================
# Quantum Inspired Annealing Solver
# =========================================================


class QuantumInspiredAnnealingSolver:


    def __init__(

            self,

            iterations=200,

            initial_temperature=10.0,

            cooling_rate=0.95,

            tunneling_decay=0.01,

            seed=42):


        """

        Parameters
        ----------


        iterations:
            number of annealing iterations


        initial_temperature:
            initial exploration temperature


        cooling_rate:
            temperature decay factor


        tunneling_decay:
            eta coefficient in psi_k


        """


        self.iterations=iterations


        self.initial_temperature=(

            initial_temperature

        )


        self.cooling_rate=cooling_rate


        self.tunneling_decay=tunneling_decay


        self.random=np.random.RandomState(

            seed

        )



        self.history=[]




    # =====================================================
    # QUBO Energy Function
    # =====================================================


    def energy(

            self,

            x,

            Q):


        """

        Compute:

        H(x)=x^TQx


        """


        return float(

            x.T @ Q @ x

        )




    # =====================================================
    # Random Neighbor Generation
    # =====================================================


    def generate_neighbor(

            self,

            x):


        """

        Generate candidate binary state.


        """


        candidate=x.copy()



        index=self.random.randint(

            len(x)

        )


        candidate[index]=1-candidate[index]



        return candidate




    # =====================================================
    # Tunneling Inspired Coefficient
    # =====================================================


    def tunneling_factor(

            self,

            iteration):


        """

        Equation:

        psi_k=e^(-eta*k)



        """


        return np.exp(

            -

            self.tunneling_decay

            *

            iteration

        )




    # =====================================================
    # Transition Probability
    # =====================================================


    def transition_probability(

            self,

            delta_energy,

            temperature,

            iteration):


        """

        Quantum-inspired transition rule.


        """


        psi=self.tunneling_factor(

            iteration

        )



        if temperature <= 1e-12:


            return 0



        probability=(

            np.exp(

                -delta_energy/

                temperature

            )

            *

            psi

        )



        return min(

            1.0,

            probability

        )




    # =====================================================
    # Temperature Update
    # =====================================================


    def update_temperature(

            self,

            temperature):


        return (

            temperature

            *

            self.cooling_rate

        )




    # =====================================================
    # Main Optimization
    # =====================================================


    def solve(

            self,

            Q,

            initial_solution=None):


        """

        Solve QUBO optimization.


        Returns:

        best binary vector

        best energy



        """


        n=Q.shape[0]



        # Initial state


        if initial_solution is None:


            current=(

                self.random.randint(

                    0,

                    2,

                    size=n

                )

            )


        else:


            current=initial_solution.copy()




        best=current.copy()



        current_energy=self.energy(

            current,

            Q

        )


        best_energy=current_energy



        temperature=(

            self.initial_temperature

        )



        self.history=[]




        # =================================================
        # Annealing Loop
        # =================================================


        for k in range(

                self.iterations):


            candidate=(

                self.generate_neighbor(

                    current

                )

            )


            candidate_energy=(

                self.energy(

                    candidate,

                    Q

                )

            )



            delta=(

                candidate_energy

                -

                current_energy

            )



            probability=(

                self.transition_probability(

                    delta,

                    temperature,

                    k

                )

            )



            # acceptance


            if (

                delta < 0

                or

                self.random.rand()

                < probability

            ):


                current=candidate


                current_energy=candidate_energy




            # update best


            if current_energy < best_energy:


                best=current.copy()


                best_energy=current_energy




            self.history.append(

                {

                "iteration":k,

                "energy":current_energy,

                "temperature":temperature,

                "tunneling":

                self.tunneling_factor(k)

                }

            )



            temperature=(

                self.update_temperature(

                    temperature

                )

            )



        return best,best_energy




    # =====================================================
    # Convergence Information
    # =====================================================


    def convergence_curve(self):


        """

        Used for Figure 9 convergence analysis.


        """


        return [

            item["energy"]

            for item in self.history

        ]




    # =====================================================
    # Reset Solver
    # =====================================================


    def reset(self):


        self.history=[]




# =========================================================
# Utility Function
# =========================================================


def solve_qubo(

        Q,

        iterations=200):


    """

    Simple interface for SEA-QAL.


    """


    solver=(

        QuantumInspiredAnnealingSolver(

            iterations=iterations

        )

    )


    solution,energy=(

        solver.solve(Q)

    )


    return solution,energy




# =========================================================
# Test Example
# =========================================================


if __name__=="__main__":


    # Example QUBO matrix


    Q=np.array(

        [

            [0.25,0.10,0.05],

            [0.10,0.40,0.08],

            [0.05,0.08,0.30]

        ]

    )



    solver=(

        QuantumInspiredAnnealingSolver(

            iterations=100

        )

    )



    solution,cost=(

        solver.solve(Q)

    )



    print(

        "Optimal Selection:",

        solution

    )


    print(

        "Minimum QUBO Energy:",

        cost

    )
