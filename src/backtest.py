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

from src.build_priors import _team_rates, _team_shots_rate
from src.schedule import remaining_fixtures
from src import dixon_coles as dc

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_ODDS_DIR = DATA_DIR / "raw_odds"

RETURNING_PRIOR_STRENGTH = 0.3
PROMOTED_PRIOR_STRENGTH = 3.0
# How much the prior mean (not the Dixon-Coles fit itself, which always
# uses real goals) leans on shots-on-target rate vs. goals rate. Grid
# searched 0-1: RPS improved monotonically to the boundary (0.2047 -> 0.2045)
# with no plausibility red flags in the resulting team values, so full
# weight on the lower-variance shots-based signal.
SHOTS_WEIGHT = 1.0


@dataclass
class TranslationClass:
    """One prior promotion class used to derive the Championship-to-PL
    translation ratio: its teams, the Championship season they were
    promoted from, and the PL season they actually played."""
    teams: list[str]
    champ_season: Path
    pl_season: Path


@dataclass
class SeasonConfig:
    target: str
    target_matches: Path
    odds: Path | None
    pl_window: list[Path]
    promoted_into_target: list[str]
    promoted_champ_season: Path
    # Multiple prior promotion classes pooled together, not just the one
    # immediately before this season — more teams means a less noisy
    # translation ratio.
    translation_classes: list[TranslationClass]
    prior_season_final: Path
    # Returning teams whose manager at the start of this season had zero
    # competitive matches in charge before it (i.e. appointed that summer,
    # not a mid-prior-season hire who already has some in-window data).
    # Verified against Wikipedia's season-manager tables; promoted teams
    # are excluded here since they're already heavily regularized toward
    # a translated prior regardless of who's managing them.
    new_manager_teams: list[str]
    # Teams playing in Champions League/Europa/Conference League that
    # season (verified via Wikipedia/Sky Sports/Premier League.com) — extra
    # midweek fixtures, used only to discount the simulated remainder of
    # the season, not the fit.
    european_teams: list[str]


SEASONS = [
    SeasonConfig(
        target="2023-24",
        target_matches=RAW_DIR / "pl_2324.csv",
        odds=RAW_ODDS_DIR / "pl_2324_odds.csv",
        pl_window=[RAW_DIR / "pl_2021.csv", RAW_DIR / "pl_2122.csv", RAW_DIR / "pl_2223.csv"],
        promoted_into_target=["Burnley", "Sheffield United", "Luton"],
        promoted_champ_season=RAW_DIR / "championship_2223.csv",
        translation_classes=[
            TranslationClass(["Fulham", "Bournemouth", "Nott'm Forest"], RAW_DIR / "championship_2122.csv", RAW_DIR / "pl_2223.csv"),
            TranslationClass(["Norwich", "Watford", "Brentford"], RAW_DIR / "championship_2021.csv", RAW_DIR / "pl_2122.csv"),
        ],
        prior_season_final=RAW_DIR / "pl_2223.csv",
        new_manager_teams=["Chelsea", "Tottenham", "Bournemouth", "Wolves"],
        european_teams=["Man City", "Arsenal", "Man United", "Newcastle", "Liverpool", "Brighton", "West Ham", "Aston Villa"],
    ),
    SeasonConfig(
        target="2024-25",
        target_matches=RAW_DIR / "pl_2425.csv",
        odds=RAW_ODDS_DIR / "pl_2425_odds.csv",
        pl_window=[RAW_DIR / "pl_2122.csv", RAW_DIR / "pl_2223.csv", RAW_DIR / "pl_2324.csv"],
        promoted_into_target=["Leicester", "Ipswich", "Southampton"],
        promoted_champ_season=RAW_DIR / "championship_2324.csv",
        translation_classes=[
            TranslationClass(["Burnley", "Sheffield United", "Luton"], RAW_DIR / "championship_2223.csv", RAW_DIR / "pl_2324.csv"),
            TranslationClass(["Fulham", "Bournemouth", "Nott'm Forest"], RAW_DIR / "championship_2122.csv", RAW_DIR / "pl_2223.csv"),
        ],
        prior_season_final=RAW_DIR / "pl_2324.csv",
        new_manager_teams=["Liverpool", "Chelsea", "Brighton", "West Ham"],
        european_teams=["Liverpool", "Arsenal", "Man City", "Chelsea", "Newcastle", "Tottenham", "Aston Villa", "Nott'm Forest", "Crystal Palace"],
    ),
    SeasonConfig(
        target="2025-26",
        target_matches=RAW_DIR / "pl_2526.csv",
        odds=RAW_ODDS_DIR / "pl_2526_odds.csv",
        pl_window=[RAW_DIR / "pl_2223.csv", RAW_DIR / "pl_2324.csv", RAW_DIR / "pl_2425.csv"],
        promoted_into_target=["Leeds", "Sunderland", "Burnley"],
        promoted_champ_season=RAW_DIR / "championship_2425.csv",
        translation_classes=[
            TranslationClass(["Leicester", "Ipswich", "Southampton"], RAW_DIR / "championship_2324.csv", RAW_DIR / "pl_2425.csv"),
            TranslationClass(["Burnley", "Sheffield United", "Luton"], RAW_DIR / "championship_2223.csv", RAW_DIR / "pl_2324.csv"),
            TranslationClass(["Fulham", "Bournemouth", "Nott'm Forest"], RAW_DIR / "championship_2122.csv", RAW_DIR / "pl_2223.csv"),
        ],
        new_manager_teams=["Brentford", "Tottenham"],
        european_teams=["Liverpool", "Arsenal", "Man City", "Chelsea", "Newcastle", "Bournemouth", "Sunderland", "Crystal Palace", "Brighton"],
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
    translation_classes=[
        TranslationClass(["Leeds", "Sunderland", "Burnley"], RAW_DIR / "championship_2425.csv", RAW_DIR / "pl_2526.csv"),
        TranslationClass(["Leicester", "Ipswich", "Southampton"], RAW_DIR / "championship_2324.csv", RAW_DIR / "pl_2425.csv"),
        TranslationClass(["Burnley", "Sheffield United", "Luton"], RAW_DIR / "championship_2223.csv", RAW_DIR / "pl_2324.csv"),
    ],
    prior_season_final=RAW_DIR / "pl_2526.csv",
    # Verified against Wikipedia + Sky Sports: 9 clubs appointed a brand-new
    # manager this summer; Ipswich is excluded here since it's already in
    # promoted_into_target (separate cold-start handling). Man Utd (Carrick,
    # permanent since 22 May 2026) and Tottenham (new manager ~March 2026)
    # are excluded too since they already have partial 2025-26 data under
    # their new manager, unlike the other 8 who have zero PL matches in
    # charge before this season's Matchweek 1.
    new_manager_teams=["Chelsea", "Man City", "Liverpool", "Newcastle", "Nott'm Forest", "Bournemouth", "Crystal Palace", "Fulham"],
    european_teams=["Arsenal", "Man City", "Man United", "Aston Villa", "Liverpool", "Bournemouth", "Sunderland", "Crystal Palace", "Brighton"],
)


def _returning_team_priors(cfg: SeasonConfig, shots_weight: float = SHOTS_WEIGHT) -> dict[str, tuple[float, float]]:
    season_rates = [_team_rates(p)[0] for p in cfg.pl_window]
    excluded = set(cfg.promoted_into_target)
    all_fd_names = set().union(*season_rates)

    if shots_weight:
        season_shots = [_team_shots_rate(p) for p in cfg.pl_window]

    priors = {}
    for fd_name in all_fd_names:
        if fd_name in excluded:
            continue
        observations = [rates[fd_name] for rates in season_rates if fd_name in rates]
        attack = sum(a for a, _ in observations) / len(observations)
        defense = sum(d for _, d in observations) / len(observations)
        if shots_weight:
            shot_obs = [rates[fd_name] for rates in season_shots if fd_name in rates]
            shot_attack = sum(a for a, _ in shot_obs) / len(shot_obs)
            shot_defense = sum(d for _, d in shot_obs) / len(shot_obs)
            attack = (1 - shots_weight) * attack + shots_weight * shot_attack
            defense = (1 - shots_weight) * defense + shots_weight * shot_defense
        priors[fd_name] = (attack, defense)
    return priors


def _pool_translation(translation_classes: list[TranslationClass], rate_fn) -> tuple[float, float, float, float]:
    """Pools every team from every translation class into one set of
    (attack_ratio, defense_ratio) observations (PL-rate / Championship-rate)
    plus the flat baseline of what all of them actually did in the PL —
    more classes means less noise in both, vs. relying on just one
    promotion class's 3 teams."""
    attack_ratios, defense_ratios, pl_attacks, pl_defenses = [], [], [], []
    for tc in translation_classes:
        champ_rates = rate_fn(tc.champ_season)
        pl_rates = rate_fn(tc.pl_season)
        for t in tc.teams:
            c_atk, c_def = champ_rates[t]
            p_atk, p_def = pl_rates[t]
            attack_ratios.append(p_atk / c_atk)
            defense_ratios.append(p_def / c_def)
            pl_attacks.append(p_atk)
            pl_defenses.append(p_def)
    attack_ratios.sort()
    defense_ratios.sort()
    mid = len(attack_ratios) // 2
    return attack_ratios[mid], defense_ratios[mid], sum(pl_attacks) / len(pl_attacks), sum(pl_defenses) / len(pl_defenses)


def _promoted_team_priors(cfg: SeasonConfig, shots_weight: float = SHOTS_WEIGHT) -> dict[str, tuple[float, float]]:
    attack_ratio, defense_ratio, baseline_attack, baseline_defense = _pool_translation(
        cfg.translation_classes, lambda p: _team_rates(p)[0]
    )
    champ_now, _ = _team_rates(cfg.promoted_champ_season)

    if shots_weight:
        attack_ratio_s, defense_ratio_s, baseline_attack_s, baseline_defense_s = _pool_translation(
            cfg.translation_classes, _team_shots_rate
        )
        champ_now_s = _team_shots_rate(cfg.promoted_champ_season)

    priors = {}
    for fd_name in cfg.promoted_into_target:
        c_atk, c_def = champ_now[fd_name]
        attack = 0.5 * baseline_attack + 0.5 * (c_atk * attack_ratio)
        defense = 0.5 * baseline_defense + 0.5 * (c_def * defense_ratio)
        if shots_weight:
            c_atk_s, c_def_s = champ_now_s[fd_name]
            attack_s = 0.5 * baseline_attack_s + 0.5 * (c_atk_s * attack_ratio_s)
            defense_s = 0.5 * baseline_defense_s + 0.5 * (c_def_s * defense_ratio_s)
            attack = (1 - shots_weight) * attack + shots_weight * attack_s
            defense = (1 - shots_weight) * defense + shots_weight * defense_s
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


def _score_remaining_matches(
    remaining: pd.DataFrame, params: dc.DixonColesParams, odds: pd.DataFrame, fatigue: dict[str, float] | None = None
) -> dict:
    params = dc.apply_fatigue(params, fatigue)
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
    new_manager_strength: float | None = None,
    half_life_days: float = dc.HALF_LIFE_DAYS,
    european_fatigue: float | None = None,
    shots_weight: float = SHOTS_WEIGHT,
) -> dict:
    target_matches = pd.read_csv(cfg.target_matches)
    mw1 = _first_round(target_matches)
    as_of = pd.to_datetime(mw1["Date"], dayfirst=True).max() + pd.Timedelta(days=1)

    prior_means = {
        **_returning_team_priors(cfg, shots_weight),
        **_promoted_team_priors(cfg, shots_weight),
    }
    prior_strength = {}
    for t in prior_means:
        if t in cfg.promoted_into_target:
            prior_strength[t] = promoted_strength
        elif new_manager_strength is not None and t in cfg.new_manager_teams:
            prior_strength[t] = new_manager_strength
        else:
            prior_strength[t] = returning_strength

    history = pd.concat([pd.read_csv(p) for p in cfg.pl_window], ignore_index=True)
    if new_manager_strength is not None and cfg.new_manager_teams:
        stale = history["HomeTeam"].isin(cfg.new_manager_teams) | history["AwayTeam"].isin(cfg.new_manager_teams)
        history = history[~stale]
    fit_pool = pd.concat([history, mw1], ignore_index=True)
    params = dc.fit(
        fit_pool, as_of, prior_means, prior_strength,
        half_life_days=half_life_days,
    )

    squads = sorted(set(target_matches["HomeTeam"]) | set(target_matches["AwayTeam"]))
    week1_table = _table_from(mw1, squads)
    fixtures = remaining_fixtures(squads)
    fatigue = None
    if european_fatigue is not None:
        fatigue = {t: european_fatigue for t in cfg.european_teams}
    predicted = dc.simulate_season(
        params, squads, fixtures,
        week1_table["Pts"].to_numpy(), week1_table["GF"].to_numpy(), week1_table["GA"].to_numpy(),
        n_sims=sims, seed=seed, fatigue=fatigue,
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
        result.update(_score_remaining_matches(remaining, params, odds, fatigue))

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
