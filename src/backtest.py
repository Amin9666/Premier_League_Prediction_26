"""Backtest the Dixon-Coles pipeline across multiple past seasons: for each
target season, fit on only the match data that would have actually existed
before its real Matchweek 1, simulate the rest of the season, and score
both (a) the simulated final table against what actually happened and (b)
match-by-match outcome probabilities against real results — benchmarked
against the closing betting-market odds for the same matches, which is the
standard "so what" comparison in football forecasting.

Testing across several seasons is the point: a model tuned to ace one
historical season is very likely just overfit to that season's noise,
since match outcomes have irreducible randomness no Week-1-only model can
eliminate. Consistent performance across seasons — and tracking, not
necessarily beating, the closing market — is the credible signal.

Run: python -m src.backtest
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.build_priors import _team_rates
from src.schedule import remaining_fixtures
from src import dixon_coles as dc

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_ODDS_DIR = DATA_DIR / "raw_odds"

RETURNING_PRIOR_STRENGTH = 0.3
PROMOTED_PRIOR_STRENGTH = 3.0


@dataclass
class SeasonConfig:
    target: str
    target_matches: Path
    odds: Path | None
    pl_window: list[Path]
    promoted_into_target: list[str]
    promoted_champ_season: Path
    translation_reference: list[str]
    translation_champ_season: Path
    translation_pl_season: Path
    prior_season_final: Path


SEASONS = [
    SeasonConfig(
        target="2023-24",
        target_matches=RAW_DIR / "pl_2324.csv",
        odds=RAW_ODDS_DIR / "pl_2324_odds.csv",
        pl_window=[RAW_DIR / "pl_2021.csv", RAW_DIR / "pl_2122.csv", RAW_DIR / "pl_2223.csv"],
        promoted_into_target=["Burnley", "Sheffield United", "Luton"],
        promoted_champ_season=RAW_DIR / "championship_2223.csv",
        translation_reference=["Fulham", "Bournemouth", "Nott'm Forest"],
        translation_champ_season=RAW_DIR / "championship_2122.csv",
        translation_pl_season=RAW_DIR / "pl_2223.csv",
        prior_season_final=RAW_DIR / "pl_2223.csv",
    ),
    SeasonConfig(
        target="2024-25",
        target_matches=RAW_DIR / "pl_2425.csv",
        odds=RAW_ODDS_DIR / "pl_2425_odds.csv",
        pl_window=[RAW_DIR / "pl_2122.csv", RAW_DIR / "pl_2223.csv", RAW_DIR / "pl_2324.csv"],
        promoted_into_target=["Leicester", "Ipswich", "Southampton"],
        promoted_champ_season=RAW_DIR / "championship_2324.csv",
        translation_reference=["Burnley", "Sheffield United", "Luton"],
        translation_champ_season=RAW_DIR / "championship_2223.csv",
        translation_pl_season=RAW_DIR / "pl_2324.csv",
        prior_season_final=RAW_DIR / "pl_2324.csv",
    ),
    SeasonConfig(
        target="2025-26",
        target_matches=RAW_DIR / "pl_2526.csv",
        odds=RAW_ODDS_DIR / "pl_2526_odds.csv",
        pl_window=[RAW_DIR / "pl_2223.csv", RAW_DIR / "pl_2324.csv", RAW_DIR / "pl_2425.csv"],
        promoted_into_target=["Leeds", "Sunderland", "Burnley"],
        promoted_champ_season=RAW_DIR / "championship_2425.csv",
        translation_reference=["Leicester", "Ipswich", "Southampton"],
        translation_champ_season=RAW_DIR / "championship_2324.csv",
        translation_pl_season=RAW_DIR / "pl_2425.csv",
        prior_season_final=RAW_DIR / "pl_2425.csv",
    ),
]

# The actual season being forecast — same shape as a backtest SeasonConfig,
# but target_matches only has however many matchweeks have been played so
# far (grows over the season; odds/prior_season_final aren't needed since
# there's no "actual final table" yet to score against).
PRODUCTION_CONFIG = SeasonConfig(
    target="2026-27",
    target_matches=RAW_DIR / "pl_2627.csv",
    odds=None,
    pl_window=[RAW_DIR / "pl_2324.csv", RAW_DIR / "pl_2425.csv", RAW_DIR / "pl_2526.csv"],
    promoted_into_target=["Coventry", "Ipswich", "Hull"],
    promoted_champ_season=RAW_DIR / "championship_2526.csv",
    translation_reference=["Leeds", "Sunderland", "Burnley"],
    translation_champ_season=RAW_DIR / "championship_2425.csv",
    translation_pl_season=RAW_DIR / "pl_2526.csv",
    prior_season_final=RAW_DIR / "pl_2526.csv",
)


def _returning_team_priors(cfg: SeasonConfig) -> dict[str, tuple[float, float]]:
    season_rates = [_team_rates(p)[0] for p in cfg.pl_window]
    excluded = set(cfg.promoted_into_target)
    all_fd_names = set().union(*season_rates)

    priors = {}
    for fd_name in all_fd_names:
        if fd_name in excluded:
            continue
        observations = [rates[fd_name] for rates in season_rates if fd_name in rates]
        priors[fd_name] = (
            sum(a for a, _ in observations) / len(observations),
            sum(d for _, d in observations) / len(observations),
        )
    return priors


def _promoted_team_priors(cfg: SeasonConfig) -> dict[str, tuple[float, float]]:
    champ_ref, _ = _team_rates(cfg.translation_champ_season)
    pl_ref, _ = _team_rates(cfg.translation_pl_season)

    attack_ratios = sorted(pl_ref[t][0] / champ_ref[t][0] for t in cfg.translation_reference)
    defense_ratios = sorted(pl_ref[t][1] / champ_ref[t][1] for t in cfg.translation_reference)
    mid = len(attack_ratios) // 2
    attack_ratio, defense_ratio = attack_ratios[mid], defense_ratios[mid]

    baseline_attack = sum(pl_ref[t][0] for t in cfg.translation_reference) / len(cfg.translation_reference)
    baseline_defense = sum(pl_ref[t][1] for t in cfg.translation_reference) / len(cfg.translation_reference)

    champ_now, _ = _team_rates(cfg.promoted_champ_season)
    priors = {}
    for fd_name in cfg.promoted_into_target:
        c_atk, c_def = champ_now[fd_name]
        attack = 0.5 * baseline_attack + 0.5 * (c_atk * attack_ratio)
        defense = 0.5 * baseline_defense + 0.5 * (c_def * defense_ratio)
        priors[fd_name] = (attack, defense)
    return priors


def _first_round(matches: pd.DataFrame) -> pd.DataFrame:
    n_teams = matches["HomeTeam"].nunique()
    seen = set()
    rows = []
    for date in sorted(matches["Date"].unique(), key=lambda d: pd.to_datetime(d, dayfirst=True)):
        for _, m in matches[matches["Date"] == date].iterrows():
            seen.add(m["HomeTeam"])
            seen.add(m["AwayTeam"])
            rows.append(m)
        if len(seen) == n_teams:
            break
    return pd.DataFrame(rows)


def _table_from(round_matches: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    rows = []
    for team in teams:
        home = round_matches[round_matches["HomeTeam"] == team]
        away = round_matches[round_matches["AwayTeam"] == team]
        mp = len(home) + len(away)
        gf = int(home["FTHG"].sum() + away["FTAG"].sum())
        ga = int(home["FTAG"].sum() + away["FTHG"].sum())
        w = int((home["FTHG"] > home["FTAG"]).sum() + (away["FTAG"] > away["FTHG"]).sum())
        d = int((home["FTHG"] == home["FTAG"]).sum() + (away["FTAG"] == away["FTHG"]).sum())
        rows.append([team, mp, w, d, mp - w - d, gf, ga, 3 * w + d])
    return pd.DataFrame(rows, columns=["Squad", "MP", "W", "D", "L", "GF", "GA", "Pts"])


def _final_table(matches: pd.DataFrame) -> pd.DataFrame:
    teams = sorted(set(matches["HomeTeam"]) | set(matches["AwayTeam"]))
    table = _table_from(matches, teams)
    table["GD"] = table["GF"] - table["GA"]
    table = table.sort_values(["Pts", "GD", "GF"], ascending=False).reset_index(drop=True)
    table.insert(0, "actual_rank", range(1, len(table) + 1))
    return table[["Squad", "actual_rank", "Pts", "GD", "GF"]]


def _rps(probs: np.ndarray, actual_idx: int) -> float:
    e = np.zeros(3)
    e[actual_idx] = 1
    return float(np.mean((np.cumsum(probs)[:-1] - np.cumsum(e)[:-1]) ** 2))


def _implied_probs(avg_h: float, avg_d: float, avg_a: float) -> np.ndarray:
    raw = np.array([1 / avg_h, 1 / avg_d, 1 / avg_a])
    return raw / raw.sum()


def _score_remaining_matches(remaining: pd.DataFrame, params: dc.DixonColesParams, odds: pd.DataFrame) -> dict:
    odds = odds[["Date", "HomeTeam", "AwayTeam", "AvgCH", "AvgCD", "AvgCA"]]
    merged = remaining.merge(odds, on=["Date", "HomeTeam", "AwayTeam"], how="inner")
    model_rps, model_ll, market_rps, market_ll = [], [], [], []
    for m in merged.itertuples():
        outcome = 0 if m.FTHG > m.FTAG else (1 if m.FTHG == m.FTAG else 2)
        p_home, p_draw, p_away = dc.match_probabilities(params, m.HomeTeam, m.AwayTeam)
        p = np.array([p_home, p_draw, p_away])
        model_rps.append(_rps(p, outcome))
        model_ll.append(-np.log(max(p[outcome], 1e-12)))

        mp = _implied_probs(m.AvgCH, m.AvgCD, m.AvgCA)
        market_rps.append(_rps(mp, outcome))
        market_ll.append(-np.log(max(mp[outcome], 1e-12)))

    return {
        "n_matches": len(merged),
        "model_rps": np.mean(model_rps),
        "model_logloss": np.mean(model_ll),
        "market_rps": np.mean(market_rps),
        "market_logloss": np.mean(market_ll),
    }


def backtest_season(
    cfg: SeasonConfig,
    sims: int = 10_000,
    seed: int = 42,
    returning_strength: float = RETURNING_PRIOR_STRENGTH,
    promoted_strength: float = PROMOTED_PRIOR_STRENGTH,
) -> dict:
    target_matches = pd.read_csv(cfg.target_matches)
    mw1 = _first_round(target_matches)
    as_of = pd.to_datetime(mw1["Date"], dayfirst=True).max() + pd.Timedelta(days=1)

    prior_means = {**_returning_team_priors(cfg), **_promoted_team_priors(cfg)}
    prior_strength = {t: (promoted_strength if t in cfg.promoted_into_target else returning_strength) for t in prior_means}

    history = pd.concat([pd.read_csv(p) for p in cfg.pl_window], ignore_index=True)
    fit_pool = pd.concat([history, mw1], ignore_index=True)
    params = dc.fit(fit_pool, as_of, prior_means, prior_strength)

    squads = sorted(set(target_matches["HomeTeam"]) | set(target_matches["AwayTeam"]))
    week1_table = _table_from(mw1, squads)
    fixtures = remaining_fixtures(squads)
    predicted = dc.simulate_season(
        params, squads, fixtures,
        week1_table["Pts"].to_numpy(), week1_table["GF"].to_numpy(), week1_table["GA"].to_numpy(),
        n_sims=sims, seed=seed,
    )

    actual = _final_table(target_matches)
    merged = predicted.merge(actual, on="Squad")
    rho = merged["avg_position"].corr(merged["actual_rank"], method="spearman")

    prior_final = _final_table(pd.read_csv(cfg.prior_season_final))
    baseline_rank = dict(zip(prior_final["Squad"], prior_final["actual_rank"]))
    champ_now, _ = _team_rates(cfg.promoted_champ_season)
    promoted_order = sorted(cfg.promoted_into_target, key=lambda t: champ_now[t][0] - champ_now[t][1], reverse=True)
    for i, team in enumerate(promoted_order):
        baseline_rank[team] = 18 + i
    merged["baseline_rank"] = merged["Squad"].map(baseline_rank)
    baseline_rho = merged["baseline_rank"].corr(merged["actual_rank"], method="spearman")

    result = {
        "target": cfg.target,
        "rho": rho,
        "baseline_rho": baseline_rho,
        "home_advantage": np.exp(params.home_advantage),
        "rho_param": params.rho,
    }

    if cfg.odds is not None:
        odds = pd.read_csv(cfg.odds)
        remaining = target_matches.merge(
            mw1[["Date", "HomeTeam", "AwayTeam"]], on=["Date", "HomeTeam", "AwayTeam"], how="left", indicator=True
        )
        remaining = remaining[remaining["_merge"] == "left_only"].drop(columns="_merge")
        result.update(_score_remaining_matches(remaining, params, odds))

    return result


def main() -> None:
    results = [backtest_season(cfg) for cfg in SEASONS]

    print("=== Backtest: predict each season's final table from its real Matchweek 1 (Dixon-Coles) ===\n")
    summary = pd.DataFrame(
        [
            {
                "Season": r["target"],
                "Rank rho": round(r["rho"], 3),
                "Baseline rho": round(r["baseline_rho"], 3),
                "Model RPS": round(r["model_rps"], 4),
                "Market RPS": round(r["market_rps"], 4),
                "Model logloss": round(r["model_logloss"], 4),
                "Market logloss": round(r["market_logloss"], 4),
                "Fitted HA": round(r["home_advantage"], 3),
                "Fitted rho": round(r["rho_param"], 4),
            }
            for r in results
        ]
    )
    print(summary.to_string(index=False))
    print()
    for key, label in [
        ("rho", "Average rank rho"),
        ("baseline_rho", "Average baseline rho"),
        ("model_rps", "Average model RPS (lower=better)"),
        ("market_rps", "Average market RPS (lower=better)"),
        ("model_logloss", "Average model log-loss (lower=better)"),
        ("market_logloss", "Average market log-loss (lower=better)"),
    ]:
        print(f"{label}: {sum(r[key] for r in results) / len(results):.4f}")


if __name__ == "__main__":
    main()
