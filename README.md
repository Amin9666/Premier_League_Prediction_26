# Premier League Standings Predictor

Predicts final 2026-27 Premier League standings from Matchweek 1 results,
using a Poisson goal-scoring model and Monte Carlo simulation of the rest
of the season.

## How it works

1. **Data (`data/week1_table.csv`)** — Matchweek 1 results/table, supplied
   by the user (this environment could not reach live sports data sites,
   so the numbers were pasted in directly).

2. **Historical priors (`data/historical_priors.csv`)** — Since one match
   is an extremely noisy signal on its own, each team's Week 1 attack/defense
   rate is blended with a prior strength tier based on general Premier
   League history through the 2024-25 season (the most recent season
   reliably known to the model): "Elite" (recent title contenders),
   "Strong" (regular top-6/European sides), "Mid-Upper", "Mid", and
   "Lower/Promoted". **These priors are approximate, qualitative estimates
   from general football knowledge, not verified live statistics** — treat
   them as reasonable starting assumptions, not ground truth.

3. **Strength ratings (`src/strength.py`)** — Team attack/defense strength
   = a shrinkage-weighted blend of Week 1 rate and historical prior
   (`SHRINKAGE_K` controls how much the single match is trusted vs. the
   prior; default treats it as worth 1 game against a 6-game-equivalent
   prior).

4. **Simulation (`src/simulate.py`)** — Generates the remaining 370
   fixtures (a full double round-robin minus the Week 1 round already
   played) and simulates each one by drawing home/away goals from a
   Poisson distribution parameterized by the two teams' attack/defense
   strengths and a fixed home-advantage factor. This is repeated for
   `--sims` full seasons (default 10,000) to build a distribution of
   outcomes per team.

5. **Output** — Average points, average final position, and probabilities
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

## Limitations

- **One match of real data is not enough to reliably predict a 38-game
  season.** This tool leans heavily on historical priors to compensate;
  results should be read as an illustrative, order-of-magnitude forecast,
  not a confident prediction. Accuracy will improve significantly once
  more matchweeks of results are added to `data/week1_table.csv` (or the
  strength model is extended to ingest multiple weeks).
- The historical priors were written from general training knowledge and
  were not cross-checked against a live database (this sandboxed
  environment cannot reach sports data sites), so treat tier assignments
  as approximate.
- The model does not account for injuries, transfers, fixture congestion,
  managerial changes, or in-season squad changes.
