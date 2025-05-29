import numpy as np
from scipy.integrate import odeint


def sir_model(y, t, contact_rate, recovery_rate):
    """SIR model differential equations.

    Args:
        y: Current state vector [susceptible, infected, recovered]
        t: Time (not used directly in equations)
        contact_rate: β - infection transmission rate
        recovery_rate: γ - recovery rate

    Returns:
        List of derivatives [dS/dt, dI/dt, dR/dt]
    """
    susceptible, infected, recovered = y
    N = susceptible + infected + recovered  # Total population

    # Prevent division by zero if N=0
    if N <= 0:
        return [0, 0, 0]

    # Differential equations
    dSdt = -contact_rate * susceptible * infected / N
    dIdt = contact_rate * susceptible * infected / N - recovery_rate * infected
    dRdt = recovery_rate * infected

    return [dSdt, dIdt, dRdt]


def simulate_disease_progression(age, initial_infected, contact_rate, recovery_rate, disease_duration, population=1000):
    """Simulate disease progression using SIR model.

    Args:
        age: Patient age (currently unused, consider incorporating)
        initial_infected: Initial number of infected individuals
        contact_rate: β - infection transmission rate
        recovery_rate: γ - recovery rate
        disease_duration: Number of days to simulate
        population: Total population size (default 1000)

    Returns:
        Tuple of (time_points, susceptible, infected, recovered)
    """
    # Input validation
    initial_infected = max(0, min(initial_infected, population))
    contact_rate = max(0, contact_rate)
    recovery_rate = max(0, recovery_rate)
    disease_duration = max(1, disease_duration)  # At least 1 day

    # Initial conditions
    susceptible_initial = population - initial_infected
    infected_initial = initial_infected
    recovered_initial = 0

    # Time points (daily intervals)
    time_points = np.linspace(0, disease_duration, disease_duration + 1)  # +1 to include start and end

    # Solve ODE
    solution = odeint(
        sir_model,
        [susceptible_initial, infected_initial, recovered_initial],
        time_points,
        args=(contact_rate, recovery_rate)
    )

    # Transpose solution
    susceptible, infected, recovered = solution.T

    return time_points, susceptible, infected, recovered