import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from datetime import datetime

# === Constants ===
DATA_PATH = "./data/historical/E0_*.csv"
CURRENT_SEASON_PATH = "./data/current/E0_2024.csv"
DECAY_FACTOR = 0.95
DEFAULT_LAMBDA = 1.5
DEFAULT_HOME_ADVANTAGE = 1.01
DRAW_THRESHOLD = 0.12
NUM_SIMULATIONS = 50000


# === Helpers ===

def parse_date(date_str):
    """
    Convert date string to datetime object, handling multiple formats.
    Returns NaT for invalid/missing formats.
    """
    if date_str in ["nan", "NaN", ""]:
        return pd.NaT
    try:
        return datetime.strptime(date_str, "%d/%m/%y")
    except ValueError:
        try:
            return datetime.strptime(date_str, "%d/%m/%Y")
        except ValueError:
            return pd.NaT


def load_and_clean_data(file_list):
    """
    Load and clean historical match data from multiple CSV files.
    Returns a concatenated DataFrame with valid date entries.
    """
    df_hist = pd.read_csv(file_list[0])
    for file in file_list[1:]:
        df = pd.read_csv(file)
        df_hist = pd.concat([df_hist, df], ignore_index=True)

    df_hist = df_hist[["HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date"]]
    df_hist["Date"] = df_hist["Date"].astype(str).apply(parse_date)
    df_hist = df_hist.dropna(subset=["Date"])
    return df_hist


def exponential_decay_weight(date, latest_date, decay_factor=DECAY_FACTOR):
    """
    Compute time decay weight for historical data based on recency.
    """
    days_since = (latest_date - date).days
    return decay_factor ** (days_since / 365)


def compute_weighted_goals(df, col_team, col_goals, latest_date):
    """
    Compute exponentially weighted average goals for each team.
    """
    team_stats = {}
    for team, group in df.groupby(col_team):
        group = group.copy()
        group["Weight"] = group["Date"].apply(lambda d: exponential_decay_weight(d, latest_date))
        weighted_goals = (group[col_goals] * group["Weight"]).sum() / group["Weight"].sum()
        team_stats[team] = weighted_goals
    return team_stats


def initialize_team_strength(home_goals, away_goals):
    """
    Calculate initial team strength as the average of smoothed home/away goals.
    """
    return {
        team: max(0.5, min(3.0, (home_goals.get(team, DEFAULT_LAMBDA) + away_goals.get(team, DEFAULT_LAMBDA)) / 2))
        for team in home_goals.keys()
    }


def simulate_goals(lam):
    """
    Simulate number of goals scored using Poisson distribution.
    """
    return np.random.poisson(lam)


def monte_carlo_win_probabilities(lambda_home, lambda_away, num_simulations=NUM_SIMULATIONS):
    """
    Run simulations to estimate win/draw/loss probabilities.
    """
    home_wins = draws = away_wins = 0
    for _ in range(num_simulations):
        hg = simulate_goals(lambda_home)
        ag = simulate_goals(lambda_away)
        if hg > ag:
            home_wins += 1
        elif hg < ag:
            away_wins += 1
        else:
            draws += 1
    return home_wins / num_simulations, draws / num_simulations, away_wins / num_simulations


def bayesian_update(prior_lambda, observed_goals, alpha=1.0):
    """
    Update team strength using Bayesian inference after each match.
    """
    return (prior_lambda * alpha + observed_goals) / (alpha + 1)


def plot_confusion_matrix(actual, predicted):
    """
    Display confusion matrix comparing predictions vs actual results.
    """
    cm = confusion_matrix(actual, predicted, labels=["H", "D", "A"])
    labels = ["Home", "Draw", "Away"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


# === Main Function ===

def main():
    # Load historical data
    file_list = sorted(glob.glob(DATA_PATH))
    if not file_list:
        raise FileNotFoundError(f"No historical CSVs found at {DATA_PATH}")
    selected_files = file_list[-10:]
    df_hist = load_and_clean_data(selected_files)

    # Compute goal averages and initial team strength
    latest_date = df_hist["Date"].max()
    home_goals = compute_weighted_goals(df_hist, "HomeTeam", "FTHG", latest_date)
    away_goals = compute_weighted_goals(df_hist, "AwayTeam", "FTAG", latest_date)
    team_strength = initialize_team_strength(home_goals, away_goals)

    # Load current season
    df = pd.read_csv(CURRENT_SEASON_PATH)
    all_teams = set(df["HomeTeam"]).union(df["AwayTeam"])
    for team in all_teams:
        if team not in team_strength:
            team_strength[team] = DEFAULT_LAMBDA

    # Predict + update loop
    sim_probs = []
    for index, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]

        # Predict BEFORE updating strength
        p_home, p_draw, p_away = monte_carlo_win_probabilities(
            team_strength[home] * DEFAULT_HOME_ADVANTAGE,
            team_strength[away] / DEFAULT_HOME_ADVANTAGE
        )
        sim_probs.append((p_home, p_draw, p_away))

        # Update AFTER observing actual goals
        team_strength[home] = bayesian_update(team_strength[home], row["FTHG"])
        team_strength[away] = bayesian_update(team_strength[away], row["FTAG"])

    # Post-processing
    df["Sim_P_Home_Win"], df["Sim_P_Draw"], df["Sim_P_Away_Win"] = zip(*sim_probs)
    df["Actual_Result"] = df.apply(
        lambda row: "H" if row["FTHG"] > row["FTAG"] else "A" if row["FTHG"] < row["FTAG"] else "D", axis=1
    )
    df["Simulated_Result"] = np.where(
        abs(df["Sim_P_Home_Win"] - df["Sim_P_Away_Win"]) < DRAW_THRESHOLD, "D",
        np.where(df["Sim_P_Home_Win"] > df["Sim_P_Away_Win"], "H", "A")
    )

    # Accuracy
    acc = accuracy_score(df["Actual_Result"], df["Simulated_Result"])
    print(f"Prediction Accuracy: {acc:.2%}")

    # Confusion Matrix
    plot_confusion_matrix(df["Actual_Result"], df["Simulated_Result"])

    # Save output (optional)
    # df.to_csv("output/predictions.csv", index=False)


if __name__ == "__main__":
    main()