"""Dixon-Coles (1997) goal-scoring model: the standard reference model in
the football-analytics literature, used here in place of a hand-blended
shrinkage average. Two things it adds over plain independent Poisson:

1. **Low-score correlation correction** (`tau`) — home and away goals in
   real matches aren't quite independent at low scores (0-0/1-1 are more
   common, 1-0/0-1 less common than independent Poisson implies). Dixon &
   Coles fit a single correlation parameter `rho` for the four low-scoring
   cells to correct this.
2. **Time-decayed maximum a posteriori (MAP) fit** — instead of manually
   blending discrete "this season" / "last 3 seasons" buckets, every team's
   attack/defense strength and the league's home-advantage and `rho` are
   fit jointly by maximum likelihood over full match history, each match
   weighted by `exp(-decay * days_ago)` so recent form matters more without
   a hand-picked season cutoff. Teams with little or no top-flight history
   (newly promoted sides) are regularized toward a prior mean (from
   `build_priors.py`'s Championship-translation estimate) with a much
   stronger penalty than data-rich teams, which is the standard way to
   handle cold-start teams in this kind of model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

RHO_SCALE = 0.3  # bounds the fitted low-score correlation to a plausible range
HALF_LIFE_DAYS = 400.0


@dataclass
class DixonColesParams:
    squads: list[str]
    attack: np.ndarray  # log-space attack strength per squad
    defense: np.ndarray  # log-space defense weakness per squad
    home_advantage: float  # log-space, single league-wide value
    rho: float
    index: dict[str, int]

    def lambdas(self, home: str, away: str) -> tuple[float, float]:
        h, a = self.index[home], self.index[away]
        lam = np.exp(self.attack[h] + self.defense[a] + self.home_advantage)
        mu = np.exp(self.attack[a] + self.defense[h])
        return lam, mu


def tau(x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    out = np.ones_like(lam)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    out[m00] = 1 - lam[m00] * mu[m00] * rho
    out[m01] = 1 + lam[m01] * rho
    out[m10] = 1 + mu[m10] * rho
    out[m11] = 1 - rho
    return out


def fit(
    matches: pd.DataFrame,
    as_of: pd.Timestamp,
    prior_means: dict[str, tuple[float, float]],
    prior_strength: dict[str, float],
    half_life_days: float = HALF_LIFE_DAYS,
) -> DixonColesParams:
    """matches: columns Date, HomeTeam, AwayTeam, FTHG, FTAG (all teams in
    matches must have an entry in prior_means/prior_strength).

    (Per-team home advantage was tried and rejected: across the whole
    tested regularization range, RPS moved only in the 4th decimal place —
    noise, not signal — so it added complexity without a robust benefit.
    Single league-wide value, fit from data rather than assumed.)"""
    squads = sorted(set(matches["HomeTeam"]) | set(matches["AwayTeam"]) | set(prior_means))
    n = len(squads)
    index = {s: i for i, s in enumerate(squads)}

    dates = pd.to_datetime(matches["Date"], dayfirst=True)
    days_ago = (as_of - dates).dt.days.to_numpy().astype(float)
    weight = np.exp(-np.log(2) / half_life_days * np.clip(days_ago, 0, None))

    home_idx = matches["HomeTeam"].map(index).to_numpy()
    away_idx = matches["AwayTeam"].map(index).to_numpy()
    x_goals = matches["FTHG"].to_numpy(dtype=float)
    y_goals = matches["FTAG"].to_numpy(dtype=float)

    prior_alpha = np.array([np.log(prior_means[s][0]) for s in squads])
    prior_beta = np.array([np.log(prior_means[s][1]) for s in squads])
    reg_weight = np.array([prior_strength.get(s, 0.3) for s in squads])

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        alpha = theta[:n]
        beta = theta[n : 2 * n]
        gamma = theta[2 * n]
        rho = RHO_SCALE * np.tanh(theta[2 * n + 1])
        return alpha, beta, gamma, rho

    def neg_log_posterior(theta: np.ndarray) -> float:
        alpha, beta, gamma, rho = unpack(theta)
        lam = np.exp(alpha[home_idx] + beta[away_idx] + gamma)
        mu = np.exp(alpha[away_idx] + beta[home_idx])

        ll = poisson.logpmf(x_goals, lam) + poisson.logpmf(y_goals, mu)
        tau_vals = np.clip(tau(x_goals, y_goals, lam, mu, rho), 1e-8, None)
        ll = ll + np.log(tau_vals)

        weighted_nll = -(weight * ll).sum()
        reg = np.sum(reg_weight * (alpha - prior_alpha) ** 2) + np.sum(reg_weight * (beta - prior_beta) ** 2)
        return weighted_nll + reg

    theta0 = np.concatenate([prior_alpha, prior_beta, [np.log(1.3)], [0.0]])
    bounds = [(-3, 3)] * n + [(-3, 3)] * n + [(0.0, 1.0)] + [(-3, 3)]

    result = minimize(neg_log_posterior, theta0, method="L-BFGS-B", bounds=bounds)
    alpha, beta, gamma, rho = unpack(result.x)
    return DixonColesParams(squads=squads, attack=alpha, defense=beta, home_advantage=gamma, rho=rho, index=index)


def apply_fatigue(params: DixonColesParams, fatigue: dict[str, float] | None) -> DixonColesParams:
    """Returns a copy of params with attack discounted / defense worsened
    for the given teams by their multiplier (< 1.0 = more fatigued). Used
    to adjust the forward-looking projection only, e.g. for teams playing
    extra midweek European fixtures — the fit itself already reflects
    real season-to-date performance as it happened."""
    if not fatigue:
        return params
    attack, defense = params.attack.copy(), params.defense.copy()
    for team, f in fatigue.items():
        i = params.index[team]
        attack[i] += np.log(f)
        defense[i] -= np.log(f)
    return DixonColesParams(
        squads=params.squads, attack=attack, defense=defense,
        home_advantage=params.home_advantage, rho=params.rho, index=params.index,
    )


def match_probabilities(params: DixonColesParams, home: str, away: str, max_goals: int = 10):
    lam, mu = params.lambdas(home, away)
    x = np.arange(max_goals + 1)
    px = poisson.pmf(x, lam)
    py = poisson.pmf(x, mu)
    grid = np.outer(px, py)

    xx, yy = np.meshgrid(x, x, indexing="ij")
    lam_grid = np.full(xx.shape, lam)
    mu_grid = np.full(xx.shape, mu)
    correction = tau(xx.ravel(), yy.ravel(), lam_grid.ravel(), mu_grid.ravel(), params.rho).reshape(xx.shape)
    grid = grid * correction
    grid = grid / grid.sum()

    # grid[x, y] = P(home scores x, away scores y); home win is x > y (lower
    # triangle), away win is y > x (upper triangle)
    p_home = np.tril(grid, k=-1).sum()
    p_draw = np.trace(grid)
    p_away = np.triu(grid, k=1).sum()
    return p_home, p_draw, p_away


def simulate_correlated_poisson(
    lam: np.ndarray, mu: np.ndarray, rho: float, rng: np.random.Generator, max_rounds: int = 6
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized rejection sampling from the tau-corrected joint
    distribution: draw independent Poisson, then accept/reject draws that
    land in the four low-score cells tau adjusts, according to tau/tau_max."""
    tau_max = 1 + RHO_SCALE  # tau is bounded within [1-RHO_SCALE, 1+RHO_SCALE] for the corrected cells
    x = rng.poisson(lam)
    y = rng.poisson(mu)
    pending = np.ones(lam.shape, dtype=bool)

    for _ in range(max_rounds):
        t = tau(x, y, lam, mu, rho)
        accept_prob = np.clip(t / tau_max, 0, 1)
        u = rng.uniform(size=lam.shape)
        accepted = u < accept_prob
        pending &= ~accepted
        if not pending.any():
            break
        x = np.where(pending, rng.poisson(lam), x)
        y = np.where(pending, rng.poisson(mu), y)
    return x, y


def simulate_season(
    params: DixonColesParams,
    squads: list[str],
    fixtures: list[tuple[str, str]],
    start_pts: np.ndarray,
    start_gf: np.ndarray,
    start_ga: np.ndarray,
    n_sims: int = 10_000,
    seed: int | None = 42,
    fatigue: dict[str, float] | None = None,
) -> pd.DataFrame:
    """squads: the current season's teams only (start_pts/gf/ga aligned to
    this list) — params.squads may be a superset, since the fit pool
    includes teams from past seasons no longer in the league.

    fatigue: optional {squad: multiplier < 1.0} applied only to the
    *simulated remaining fixtures*, not the fit — e.g. for teams playing
    extra midweek European games. The fit already reflects a team's real
    season-to-date performance (including any fatigue that already
    happened); this only discounts the *forward-looking* projection."""
    rng = np.random.default_rng(seed)
    n = len(squads)
    local_idx = {s: i for i, s in enumerate(squads)}

    home_idx = np.array([local_idx[h] for h, _ in fixtures])
    away_idx = np.array([local_idx[a] for _, a in fixtures])
    home_fit = np.array([params.index[h] for h, _ in fixtures])
    away_fit = np.array([params.index[a] for _, a in fixtures])

    params = apply_fatigue(params, fatigue)
    lam = np.exp(params.attack[home_fit] + params.defense[away_fit] + params.home_advantage)
    mu = np.exp(params.attack[away_fit] + params.defense[home_fit])

    final_points = np.zeros((n_sims, n), dtype=np.int32)
    final_gf = np.zeros((n_sims, n), dtype=np.int32)
    final_ga = np.zeros((n_sims, n), dtype=np.int32)
    final_rank = np.zeros((n_sims, n), dtype=np.int32)

    for s in range(n_sims):
        home_goals, away_goals = simulate_correlated_poisson(lam, mu, params.rho, rng)

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
