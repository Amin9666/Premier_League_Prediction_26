"""Generate a double round-robin fixture list for the remaining season.

Round 1 of the round-robin stands in for the already-played Week 1 (whose
results are taken from the actual data, not simulated), so only rounds
2..38 are simulated -- 37 remaining matches per team.
"""
from itertools import combinations


def remaining_fixtures(squads: list[str]) -> list[tuple[str, str]]:
    n = len(squads)
    assert n % 2 == 0, "circle method requires an even number of teams"

    teams = list(squads)
    fixed = teams[0]
    rotating = teams[1:]

    rounds = []
    for _ in range(n - 1):
        order = [fixed] + rotating
        pairs = [(order[i], order[n - 1 - i]) for i in range(n // 2)]
        rounds.append(pairs)
        rotating = rotating[1:] + rotating[:1]

    # First half: rounds as generated (round 0 stands in for Week 1, skip it).
    # Second half: same pairings with venue swapped.
    fixtures = []
    for home, away in [pair for r in rounds[1:] for pair in r]:
        fixtures.append((home, away))
    for home, away in [pair for r in rounds[1:] for pair in r]:
        fixtures.append((away, home))
    for home, away in rounds[0]:
        fixtures.append((away, home))  # reverse fixture of the Week 1 round

    # Full double round-robin = n*(n-1) games; Week 1 (n/2 games) already played.
    expected = n * (n - 1) - n // 2
    assert len(fixtures) == expected, (len(fixtures), expected)
    return fixtures
