#!/usr/bin/env python3
"""Predict Premier League final standings using a Dixon-Coles goal-scoring
model (see src/dixon_coles.py), fit on real match data: the last 3
completed PL seasons plus this season's actual results so far, all
time-decay weighted, with newly promoted teams regularized toward a
Championship-translated prior. See README.md for the full methodology and
src/backtest.py for out-of-sample validation.

Usage:
    python main.py [--sims 10000] [--seed 42] [--out predicted_table.csv]
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.backtest import (
    PRODUCTION_CONFIG,
    PROMOTED_PRIOR_STRENGTH,
    RETURNING_PRIOR_STRENGTH,
    _promoted_team_priors,
    _returning_team_priors,
    _table_from,
)
from src.build_priors import NAME_MAP
from src.schedule import remaining_fixtures
from src import dixon_coles as dc


def build_forecast(sims: int, seed: int | None) -> pd.DataFrame:
    cfg = PRODUCTION_CONFIG
    played = pd.read_csv(cfg.target_matches)
    as_of = pd.to_datetime(played["Date"], dayfirst=True).max() + pd.Timedelta(days=1)

    prior_means = {**_returning_team_priors(cfg), **_promoted_team_priors(cfg)}
    prior_strength = {
        team: (PROMOTED_PRIOR_STRENGTH if team in cfg.promoted_into_target else RETURNING_PRIOR_STRENGTH)
        for team in prior_means
    }

    history = pd.concat([pd.read_csv(p) for p in cfg.pl_window], ignore_index=True)
    fit_pool = pd.concat([history, played], ignore_index=True)
    params = dc.fit(fit_pool, as_of, prior_means, prior_strength)

    squads = sorted(set(played["HomeTeam"]) | set(played["AwayTeam"]))
    standings = _table_from(played, squads)
    fixtures = remaining_fixtures(squads)

    result = dc.simulate_season(
        params,
        squads,
        fixtures,
        standings["Pts"].to_numpy(),
        standings["GF"].to_numpy(),
        standings["GA"].to_numpy(),
        n_sims=sims,
        seed=seed,
    )
    result["Squad"] = result["Squad"].map(NAME_MAP)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sims", type=int, default=10_000, help="number of Monte Carlo season simulations")
    parser.add_argument("--seed", type=int, default=42, help="random seed (use -1 for a random seed)")
    parser.add_argument("--out", type=str, default="predicted_table.csv", help="output CSV path")
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed
    result = build_forecast(args.sims, seed)

    pd_options = ["Predicted Rank", "Squad", "avg_points", "avg_position", "title_pct", "top4_pct", "relegation_pct"]
    display = result[pd_options].round(1)
    display.columns = ["Rank", "Squad", "Avg Pts", "Avg Pos", "Title %", "Top 4 %", "Relegation %"]

    print(f"\nPredicted final Premier League table ({args.sims:,} simulated seasons, Dixon-Coles model):\n")
    print(display.to_string(index=False))

    result.to_csv(args.out, index=False)
    print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
