#!/usr/bin/env python3
"""Predict Premier League final standings from Week 1 results.

Usage:
    python main.py [--sims 10000] [--seed 42] [--out predicted_table.csv]
"""
import argparse

from src.data_loader import load_teams
from src.strength import compute_strengths
from src.simulate import simulate_season


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sims", type=int, default=10_000, help="number of Monte Carlo season simulations")
    parser.add_argument("--seed", type=int, default=42, help="random seed (use -1 for a random seed)")
    parser.add_argument("--out", type=str, default="predicted_table.csv", help="output CSV path")
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed

    teams = load_teams()
    teams = compute_strengths(teams)
    result = simulate_season(teams, n_sims=args.sims, seed=seed)

    pd_options = ["Predicted Rank", "Squad", "avg_points", "avg_position", "title_pct", "top4_pct", "relegation_pct"]
    display = result[pd_options].round(1)
    display.columns = ["Rank", "Squad", "Avg Pts", "Avg Pos", "Title %", "Top 4 %", "Relegation %"]

    print(f"\nPredicted final Premier League table ({args.sims:,} simulated seasons, from Week 1 data):\n")
    print(display.to_string(index=False))

    result.to_csv(args.out, index=False)
    print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
