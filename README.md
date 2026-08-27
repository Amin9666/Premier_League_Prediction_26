# Premier League Standings Predictor

Predicts final 2026-27 Premier League standings from real match data, using
a **Dixon-Coles** goal-scoring model (Dixon & Coles, 1997 — the reference
model in the football-forecasting literature) and Monte Carlo simulation of
the rest of the season.

## How it works

1. **Match data (`data/raw/`)** — Real match results from
   football-data.co.uk, including shots-on-target (`HST`/`AST`): the last
   3 completed Premier League seasons (2023-24, 2024-25, 2025-26), this
   season's actual results so far (`pl_2627.csv`), and several seasons of
   Championship results (used to seed newly promoted teams — see below).

2. **Cold-start priors (`src/backtest.py`: `_returning_team_priors` /
   `_promoted_team_priors`, built on the same logic as `src/build_priors.py`)**
   — every team needs a starting point for the model fit below, blending
   goals-based and shots-on-target-based rate (fully shots-weighted —
   shots are a lower-variance signal than the few goals a team actually
   scores, see "Attempted improvements"):
   - **Returning teams**: their average rate across whichever of the last
     3 PL seasons they played in, each season normalized by its own
     league-average.
   - **Newly promoted teams** (Coventry City, Ipswich Town, Hull City for
     2026-27): their Championship rate translated into PL-equivalent terms
     using a ratio pooled across the last 3 promotion classes (9 teams,
     not just the 3 immediately prior) — how each actually performed in
     the PL vs. their real Championship rate — averaged 50/50 with the
     flat baseline of what those classes actually achieved in the PL.

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
| 2024-25 | 0.720 | 0.689 | 0.211 | 0.197 | 1.013 | 0.971 |
| 2025-26 | 0.591 | 0.573 | 0.211 | 0.206 | 1.033 | 1.015 |
| **Average** | **0.716** | **0.692** | **0.205** | **0.195** | **0.991** | **0.964** |

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

## Attempted improvements

Nine follow-up ideas were tried after the initial build, each held to the
same bar: real data, calibrated or checked against the 3-season backtest,
kept only if it actually helped. Most didn't — that's expected and worth
reporting honestly rather than quietly dropping the ones that failed.

| Idea | Outcome |
|---|---|
| Shots-on-target rate (`HST`/`AST`) blended into the prior mean, instead of pure goals rate | **Kept.** A lower-variance proxy for chance quality than actual goals (finishing luck evens out over more shots than the few goals a team scores). RPS improved monotonically from 0.2047 to 0.2045 as the blend weight increased toward pure shots-based, with no plausibility issues in the resulting values — full weight adopted (`SHOTS_WEIGHT = 1.0` in `src/backtest.py`). Only changes the prior anchor; the Dixon-Coles fit itself still always uses real goals. |
| Widen the promoted-team translation sample from 1 prior promotion class (3 teams) to 2-3 classes (6-9 teams) | **Kept.** Pooling more promotion classes (fetched one more Championship season, 2020-21, to make this possible for the earliest backtest year) reduced noise in the translation ratio — rank correlation improved from 0.711 to 0.716 average, RPS held or improved in every season, no downside found. |
| Grid-search the time-decay half-life (`HALF_LIFE_DAYS`) | **Confirmed, not changed.** 400 days already ties for best RPS across a 150–1000 day sweep and has the best rank correlation among the tied group — validated rather than just assumed. |
| Per-team home advantage (instead of one league-wide constant) | **Tried, reverted.** Across the whole regularization range tested, RPS moved only in the 4th decimal place (0.2043–0.2052) — noise, not signal — while thin-data teams (e.g. a newly promoted side with one home match) produced implausible values (a 1.79x home multiplier from a single game) unless regularized so hard it collapsed back to ≈ the single global value anyway. Added complexity, no robust benefit. |
| Managerial changes (9 clubs got a brand-new manager this summer, including Man City losing Guardiola and Liverpool losing Slot) | **Tried, rejected.** Discounting those teams' pre-change historical data, calibrated against the equivalent events in each backtest season (4 in 2023-24, 4 in 2024-25, 2 in 2025-26), made predictions monotonically worse — RPS rose from 0.205 to 0.233 as the discount got more aggressive, rank correlation fell from 0.71 to 0.60. Squad talent (already captured by the multi-season match data) apparently outweighs manager identity for this purpose. |
| European fixture congestion (9 clubs in Champions League/Europa/Conference this season) | **Tried, rejected.** Same pattern: discounting European-competition teams' simulated remaining fixtures, calibrated the same way, made RPS and rank correlation worse monotonically at every strength tested. Teams good enough to qualify for Europe are already correctly rated strong by their match history; a blanket discount just under-rates them. |
| This summer's real transfer net-spend (all 20 clubs, from Transfermarkt) | **Dropped before calibration.** Needed reliable historical net-spend for the 3 backtest summers to validate against; Transfermarkt's date-filtered/historical views returned identical figures regardless of the requested season, and web.archive.org (the fallback for genuine historical snapshots) is blocked entirely in this environment — no reliable ground truth to calibrate against. |
| Squad market value as a more stable alternative to net-spend | **Dropped, same reason.** Historical "as of" snapshots returned the current squad values verbatim regardless of the requested date; the Wayback Machine workaround was also blocked. |
| Expected goals (xG) instead of actual goals for attack/defense | **Infeasible.** Understat's tables are JavaScript-rendered (not visible to a static fetch); FBref and FootballCritic both block the request outright (403). No accessible source found. |

## Limitations

- **Real accuracy gains from here need new signal**, not more tuning of
  existing parameters — the nine attempts above show most of the cheap
  wins are exhausted. xG and reliable historical transfer/squad value data
  are the most likely real levers if a source becomes reachable;
  incorporating betting-market odds directly (available in `data/raw_odds/`
  for the backtest seasons, not currently used as a model input) would also
  help but stops being "beats the market without using market data."
- No injuries, fixture-by-fixture fatigue, or in-season squad changes.
- Home advantage and `rho` are fit as single league-wide constants.
