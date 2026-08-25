"""Monte Carlo season simulation using a Poisson goal model."""
import numpy as np
import pandas as pd

from .schedule import remaining_fixtures

HOME_ADVANTAGE = 1.15
AWAY_DISADVANTAGE = 0.87


def simulate_season(
    teams: pd.DataFrame,
    n_sims: int = 10_000,
    seed: int | None = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    squads = teams["Squad"].tolist()
    idx = {s: i for i, s in enumerate(squads)}
    n = len(squads)

    attack = teams["attack"].to_numpy()
    defense = teams["defense"].to_numpy()
    league_avg = teams["league_avg_goals"].iloc[0]

    start_pts = teams["Pts"].to_numpy()
    start_gf = teams["GF"].to_numpy()
    start_ga = teams["GA"].to_numpy()

    fixtures = remaining_fixtures(squads)
    home_idx = np.array([idx[h] for h, _ in fixtures])
    away_idx = np.array([idx[a] for _, a in fixtures])

    home_lambda = league_avg * attack[home_idx] * defense[away_idx] * HOME_ADVANTAGE
    away_lambda = league_avg * attack[away_idx] * defense[home_idx] * AWAY_DISADVANTAGE

    final_points = np.zeros((n_sims, n), dtype=np.int32)
    final_gf = np.zeros((n_sims, n), dtype=np.int32)
    final_ga = np.zeros((n_sims, n), dtype=np.int32)
    final_rank = np.zeros((n_sims, n), dtype=np.int32)

    for s in range(n_sims):
        home_goals = rng.poisson(home_lambda)
        away_goals = rng.poisson(away_lambda)

        pts = start_pts.copy()
        gf = start_gf.copy()
        ga = start_ga.copy()

        np.add.at(gf, home_idx, home_goals)
        np.add.at(ga, home_idx, away_goals)
        np.add.at(gf, away_idx, away_goals)
        np.add.at(ga, away_idx, home_goals)

        home_win = home_goals > away_goals
        away_win = away_goals > home_goals
        draw = home_goals == away_goals

        np.add.at(pts, home_idx, 3 * home_win + draw)
        np.add.at(pts, away_idx, 3 * away_win + draw)

        final_points[s] = pts
        final_gf[s] = gf
        final_ga[s] = ga

        gd = gf - ga
        order = np.lexsort((-gf, -gd, -pts))
        ranks = np.empty(n, dtype=np.int32)
        ranks[order] = np.arange(1, n + 1)
        final_rank[s] = ranks

    result = pd.DataFrame(
        {
            "Squad": squads,
            "avg_points": final_points.mean(axis=0),
            "avg_position": final_rank.mean(axis=0),
            "title_pct": (final_rank == 1).mean(axis=0) * 100,
            "top4_pct": (final_rank <= 4).mean(axis=0) * 100,
            "relegation_pct": (final_rank >= n - 2).mean(axis=0) * 100,
        }
    ).sort_values("avg_position").reset_index(drop=True)
    result.insert(0, "Predicted Rank", range(1, n + 1))
    return result
