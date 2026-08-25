"""Load Week 1 results and historical strength priors."""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_week1(path: Path = DATA_DIR / "week1_table.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = ["Squad", "MP", "W", "D", "L", "GF", "GA", "Pts"]
    return df[keep].copy()


def load_priors(path: Path = DATA_DIR / "historical_priors.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[["Squad", "PriorAttack", "PriorDefense"]].copy()


def load_teams(
    week1_path: Path = DATA_DIR / "week1_table.csv",
    priors_path: Path = DATA_DIR / "historical_priors.csv",
) -> pd.DataFrame:
    week1 = load_week1(week1_path)
    priors = load_priors(priors_path)
    teams = week1.merge(priors, on="Squad", how="left")
    if teams["PriorAttack"].isna().any():
        missing = teams.loc[teams["PriorAttack"].isna(), "Squad"].tolist()
        raise ValueError(f"No historical prior found for: {missing}")
    return teams
