"""Build data/historical_priors.csv from real historical match results.

Replaces hand-typed, qualitative strength tiers with priors computed from
actual final-table goals-for/against, so the numbers `strength.py` blends
with Week 1 results are grounded in real match data rather than general
football knowledge.

Two groups of teams are handled differently, since Premier League history
isn't available for teams that weren't in the Premier League:

- **Returning teams** (played in the PL in one or more of the last three
  seasons): a recency-weighted average of their attack/defense rate in
  each of those seasons, each rate normalized by that season's own
  league-average goals/game so differing scoring environments don't bias
  the blend. Weights: 2025-26 x0.5, 2024-25 x0.3, 2023-24 x0.2 (renormalized
  over whichever seasons a team actually has data for).

- **Newly promoted teams** (no PL data in the last three seasons): their
  Championship rate is translated into PL-equivalent terms using a
  translation ratio empirically derived from the previous promotion class
  (Leeds/Sunderland/Burnley: their actual 2024-25 Championship rates vs.
  their actual 2025-26 Premier League rates), then averaged 50/50 with the
  flat baseline of what that same previous promotion class actually
  achieved in the PL. This keeps each newly promoted team differentiated
  by its own Championship form while regressing hard toward what promoted
  teams have realistically done at this level.

Source: match-result CSVs from football-data.co.uk, trimmed to
Date/HomeTeam/AwayTeam/FTHG/FTAG and stored under data/raw/.

Run: python -m src.build_priors
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

# football-data.co.uk short names -> Squad names used elsewhere in this repo.
NAME_MAP = {
    "Man City": "Manchester City",
    "Man United": "Manchester Utd",
    "Newcastle": "Newcastle",
    "Nott'm Forest": "Nottingham",
    "Tottenham": "Tottenham",
    "Leeds": "Leeds United",
    "Sunderland": "Sunderland",
    "Hull": "Hull City",
    "Ipswich": "Ipswich Town",
    "Coventry": "Coventry City",
    "Crystal Palace": "Crystal Palace",
    "Bournemouth": "Bournemouth",
    "Brighton": "Brighton",
    "Brentford": "Brentford",
    "Aston Villa": "Aston Villa",
    "Arsenal": "Arsenal",
    "Chelsea": "Chelsea",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Liverpool": "Liverpool",
    "West Ham": "West Ham",
    "Wolves": "Wolves",
    "Burnley": "Burnley",
}

# 2026-27 promotion class and the class they replace as the empirical
# translation reference (last season's promoted teams, whose actual PL
# performance is now known).
PROMOTED_THIS_SEASON = ["Coventry", "Ipswich", "Hull"]
PROMOTED_LAST_SEASON = ["Leeds", "Sunderland", "Burnley"]

# 2025-26 Premier League relegated the reference class's PL data comes from;
# 2024-25 Championship is where that class was promoted from.
PL_SEASONS = {
    "2023-24": (RAW_DIR / "pl_2324.csv", 0.2),
    "2024-25": (RAW_DIR / "pl_2425.csv", 0.3),
    "2025-26": (RAW_DIR / "pl_2526.csv", 0.5),
}
CHAMPIONSHIP_REFERENCE_SEASON = RAW_DIR / "championship_2425.csv"
CHAMPIONSHIP_CURRENT_SEASON = RAW_DIR / "championship_2526.csv"
PL_REFERENCE_SEASON = RAW_DIR / "pl_2526.csv"


def _team_rates(path: Path) -> tuple[dict[str, tuple[float, float]], float]:
    """Per-team (attack_rate, defense_rate) normalized by that match set's
    own average goals/game, plus the league average itself."""
    df = pd.read_csv(path)
    league_avg = (df["FTHG"].sum() + df["FTAG"].sum()) / (2 * len(df))

    rates = {}
    for team in sorted(set(df["HomeTeam"]) | set(df["AwayTeam"])):
        home = df[df["HomeTeam"] == team]
        away = df[df["AwayTeam"] == team]
        mp = len(home) + len(away)
        gf = home["FTHG"].sum() + away["FTAG"].sum()
        ga = home["FTAG"].sum() + away["FTHG"].sum()
        rates[team] = (gf / mp / league_avg, ga / mp / league_avg)
    return rates, league_avg


def _returning_team_priors() -> dict[str, tuple[float, float]]:
    season_rates = {season: _team_rates(path)[0] for season, (path, _weight) in PL_SEASONS.items()}
    promoted_fd_names = set(PROMOTED_THIS_SEASON)

    priors = {}
    for fd_name, squad in NAME_MAP.items():
        if fd_name in promoted_fd_names or fd_name in ("West Ham", "Wolves", "Burnley"):
            continue  # relegated in 2025-26 or newly promoted now; handled separately/excluded
        observations = []
        for season, (_, weight) in PL_SEASONS.items():
            rate = season_rates[season].get(fd_name)
            if rate is not None:
                observations.append((*rate, weight))
        weight_sum = sum(w for _, _, w in observations)
        attack = sum(a * w for a, _, w in observations) / weight_sum
        defense = sum(d * w for _, d, w in observations) / weight_sum
        priors[squad] = (attack, defense, len(observations))
    return priors


def _promotion_translation_ratio() -> tuple[float, float]:
    """Median (attack, defense) ratio of actual-PL-rate / Championship-rate
    for last season's promotion class, translating Championship form into
    PL-equivalent terms."""
    champ_rates, _ = _team_rates(CHAMPIONSHIP_REFERENCE_SEASON)
    pl_rates, _ = _team_rates(PL_REFERENCE_SEASON)

    attack_ratios, defense_ratios = [], []
    for team in PROMOTED_LAST_SEASON:
        c_attack, c_defense = champ_rates[team]
        p_attack, p_defense = pl_rates[team]
        attack_ratios.append(p_attack / c_attack)
        defense_ratios.append(p_defense / c_defense)

    attack_ratios.sort()
    defense_ratios.sort()
    mid = len(attack_ratios) // 2
    return attack_ratios[mid], defense_ratios[mid]


def _promoted_team_priors() -> dict[str, tuple[float, float]]:
    attack_ratio, defense_ratio = _promotion_translation_ratio()

    pl_rates, _ = _team_rates(PL_REFERENCE_SEASON)
    baseline_attack = sum(pl_rates[t][0] for t in PROMOTED_LAST_SEASON) / len(PROMOTED_LAST_SEASON)
    baseline_defense = sum(pl_rates[t][1] for t in PROMOTED_LAST_SEASON) / len(PROMOTED_LAST_SEASON)

    champ_rates, _ = _team_rates(CHAMPIONSHIP_CURRENT_SEASON)

    priors = {}
    for fd_name in PROMOTED_THIS_SEASON:
        c_attack, c_defense = champ_rates[fd_name]
        translated_attack = c_attack * attack_ratio
        translated_defense = c_defense * defense_ratio
        attack = 0.5 * baseline_attack + 0.5 * translated_attack
        defense = 0.5 * baseline_defense + 0.5 * translated_defense
        priors[NAME_MAP[fd_name]] = (attack, defense, 0)
    return priors


def _tier_label(attack: float, defense: float, ranked_composites: list[float], composite: float) -> str:
    rank = ranked_composites.index(composite)
    n = len(ranked_composites)
    fraction = rank / n
    if fraction < 0.2:
        return "Elite"
    if fraction < 0.4:
        return "Strong"
    if fraction < 0.6:
        return "Mid-Upper"
    if fraction < 0.8:
        return "Mid"
    return "Lower/Promoted"


def build_priors() -> pd.DataFrame:
    returning = _returning_team_priors()
    promoted = _promoted_team_priors()
    all_priors = {**returning, **promoted}

    composites = sorted((defense - attack for attack, defense, _n in all_priors.values()))

    rows = []
    for squad, (attack, defense, n_seasons) in all_priors.items():
        composite = defense - attack
        tier = _tier_label(attack, defense, composites, composite)
        if n_seasons == 0:
            basis = "Promoted for 2026-27; Championship 2025-26 form translated to PL-equivalent rates"
        else:
            basis = f"Recency-weighted average of {n_seasons} PL season(s) through 2025-26"
        rows.append([squad, tier, round(attack, 3), round(defense, 3), basis])

    df = pd.DataFrame(rows, columns=["Squad", "Tier", "PriorAttack", "PriorDefense", "Basis"])
    return df.sort_values("PriorAttack", ascending=False).reset_index(drop=True)


def main() -> None:
    df = build_priors()
    out_path = DATA_DIR / "historical_priors.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} team priors to {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
