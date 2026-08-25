"""Derive attack/defense strength ratings from Week 1 results + historical priors."""
import pandas as pd

# How many "games" of weight the historical prior carries relative to the
# single observed Week 1 match. Higher = trust history more over one match.
SHRINKAGE_K = 6.0


def compute_strengths(teams: pd.DataFrame, k: float = SHRINKAGE_K) -> pd.DataFrame:
    teams = teams.copy()
    league_avg_goals = teams["GF"].sum() / teams["MP"].sum()

    teams["week1_attack"] = teams["GF"] / teams["MP"] / league_avg_goals
    teams["week1_defense"] = teams["GA"] / teams["MP"] / league_avg_goals

    games = teams["MP"]
    teams["attack"] = (games * teams["week1_attack"] + k * teams["PriorAttack"]) / (games + k)
    teams["defense"] = (games * teams["week1_defense"] + k * teams["PriorDefense"]) / (games + k)

    teams["league_avg_goals"] = league_avg_goals
    return teams
