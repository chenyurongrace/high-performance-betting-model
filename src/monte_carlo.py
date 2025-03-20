import numpy as np

# Set the number of simulated matches
NUM_SIMULATIONS = 10000

# Define average goals per match for Team A and Team B
lambda_teamA = 1.5  # Average goals per match for Team A
lambda_teamB = 1.2  # Average goals per match for Team B

# Track outcomes
teamA_wins = 0
teamB_wins = 0
draws = 0

# Monte Carlo Simulation
for _ in range(NUM_SIMULATIONS):
    goals_A = np.random.poisson(lambda_teamA)  # Simulating goals for Team A
    goals_B = np.random.poisson(lambda_teamB)  # Simulating goals for Team B

    if goals_A > goals_B:
        teamA_wins += 1
    elif goals_B > goals_A:
        teamB_wins += 1
    else:
        draws += 1

# Calculate probabilities
p_teamA_win = teamA_wins / NUM_SIMULATIONS
p_teamB_win = teamB_wins / NUM_SIMULATIONS
p_draw = draws / NUM_SIMULATIONS

# Print results
print(f"Monte Carlo Simulation Results ({NUM_SIMULATIONS} matches)")
print(f"Team A Win Probability: {p_teamA_win:.2%}")
print(f"Team B Win Probability: {p_teamB_win:.2%}")
print(f"Draw Probability: {p_draw:.2%}")