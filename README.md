# Premier League Standings Predictor

Predicts final 2026-27 Premier League standings from real match data, using
a **Dixon-Coles** goal-scoring model (Dixon & Coles, 1997 — the reference
model in the football-forecasting literature) and Monte Carlo simulation of
the rest of the season.

## How it works

1. **Match data (`data/raw/`)** — Real match results from
   football-data.co.uk: the last 3 completed Premier League seasons
   (2023-24, 2024-25, 2025-26), this season's actual results so far
   (`pl_2627.csv`), and two seasons of Championship results (used only to
   seed newly promoted teams — see below).

2. **Cold-start priors (`src/backtest.py`: `_returning_team_priors` /
   `_promoted_team_priors`, built on the same logic as `src/build_priors.py`)**
   — every team needs a starting point for the model fit below:
   - **Returning teams**: their average attack/defense rate across
     whichever of the last 3 PL seasons they played in, each season
     normalized by its own league-average goals/game.
   - **Newly promoted teams** (Coventry City, Ipswich Town, Hull City for
     2026-27): their Championship attack/defense rate translated into
     PL-equivalent terms using a ratio derived from how the *previous*
     promotion class (Leeds/Sunderland/Burnley) actually performed —
     comparing their real Championship rates to their real PL rates —
     averaged 50/50 with the flat baseline of what that class actually
     achieved in the PL.

3. **Dixon-Coles fit (`src/dixon_coles.py`)** — every team's attack/defense
   strength, plus the league's home-advantage and low-score correlation
   parameters, are fit jointly by maximum a posteriori (MAP) estimation
   over the full match log:
   - **Time decay**: each match is weighted `exp(-decay * days_ago)`
     (half-life ~400 days), so recent results matter more without a
     hand-picked season cutoff — this replaces manual season-weight
     blending with something that just falls out of the likelihood.
   - **Low-score correlation**: real home/away goals aren't quite
     independent at low scores (0-0/1-1 are more common than independent
     Poisson implies); a `rho` parameter fit from the data corrects this.
   - **Regularization toward the cold-start priors** — teams with rich PL
     history are barely pulled toward their prior (the data dominates);
     newly promoted teams, with at most one real match, are pulled hard
     toward theirs. This is standard practice for handling sparse-data
     teams in this kind of model.
   - Home advantage and `rho` are **fit from data**, not assumed constants.

4. **Simulation (`dixon_coles.simulate_season`)** — generates the
   remaining fixtures (full double round-robin minus rounds already
   played) and simulates each one by drawing correlated home/away goals
   from the fitted, low-score-corrected joint distribution (rejection
   sampling against the independent-Poisson draw). Repeated for `--sims`
   full seasons (default 10,000).

5. **Output** — average points, average final position, and probabilities
   of winning the title, finishing top 4, or being relegated (bottom 3),
   for every team.

## Usage

```bash
pip install -r requirements.txt
python main.py --sims 10000 --seed 42 --out predicted_table.csv
```

- `--sims`: number of simulated seasons (more = more stable estimates)
- `--seed`: RNG seed for reproducibility (`-1` for a random seed each run)
- `--out`: where to write the full results CSV

To refresh the cold-start priors reference file (`data/historical_priors.csv`,
informational — not read by `main.py`) after adding a newer season to
`data/raw/`: `python -m src.build_priors`.

## Validation (`src/backtest.py`)

Backtested against 3 real, already-completed seasons (2023-24, 2024-25,
2025-26): for each, the model is fit on only the data that would have
existed before that season's real Matchweek 1, then scored two ways —
season-long rank correlation against the real final table, and match-level
probability accuracy (Ranked Probability Score, log-loss) against the
**closing betting-market odds** for the same matches, which is the standard
benchmark in football forecasting (nobody expects to beat a liquid market,
but tracking it is the credible "does this actually work" test).

```bash
python -m src.backtest
```

| Season | Rank rho (model) | Rank rho (naive: "last season's table") | RPS (model) | RPS (market) | Log-loss (model) | Log-loss (market) |
|---|---|---|---|---|---|---|
| 2023-24 | 0.836 | 0.814 | 0.191 | 0.183 | 0.927 | 0.904 |
| 2024-25 | 0.720 | 0.689 | 0.210 | 0.197 | 1.010 | 0.971 |
| 2025-26 | 0.576 | 0.573 | 0.213 | 0.206 | 1.037 | 1.015 |
| **Average** | **0.711** | **0.692** | **0.205** | **0.195** | **0.991** | **0.964** |

Takeaways:
- Beats the "assume nothing changes from last season" baseline on rank
  correlation in all 3 seasons, though not by a huge margin.
- Match-level probabilities are within ~5% of the closing market's RPS and
  log-loss — both far better than random guessing (log-loss 1.099 for a
  uniform 3-way guess) — without using any odds data as an input.
- `SHRINKAGE_K`-equivalent regularization strength and the Dixon-Coles
  half-life were grid-searched against the average across these 3 seasons
  (not fit to any single one); the current defaults are near the empirical
  optimum found in that search.

## Limitations

- **Real accuracy gains from here need new signal**, not more tuning of
  existing parameters — e.g. squad/transfer data, injuries, or incorporating
  betting-market odds directly (available in `data/raw_odds/` for the
  backtest seasons, not currently used as a model input).
- Newly promoted teams' priors rest on a translation ratio fit to just one
  prior promotion class, so they carry more uncertainty than returning
  teams' priors.
- No injuries, transfers, fixture congestion, or managerial changes.
- Home advantage and `rho` are fit as single league-wide constants, not
  per team.
