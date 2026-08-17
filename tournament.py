"""Create a six-game tournament schedule from a Team,Division CSV.

The division-play format is five teams per division:
one exhibition round, followed by five division rounds.  In the division
rounds each team plays every other team once and has one bye.
"""

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TournamentTeam:
    name: str
    division: str


@dataclass(frozen=True)
class Match:
    game: int
    match_number: int
    team_1: str
    team_2: str = "BYE"
    division: str = ""
    match_type: str = "Division"


def _normalise_header(value: str) -> str:
    return " ".join(value.strip().lower().split())


def load_tournament_teams(csv_file: str | Path) -> list[TournamentTeam]:
    """Load teams from a CSV containing exactly the required Team/Division fields."""
    with open(csv_file, newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("CSV must have a header row: Team,Division")

        columns = {_normalise_header(name): name for name in reader.fieldnames}
        missing = [name for name in ("team", "division") if name not in columns]
        if missing:
            raise ValueError(
                "CSV is missing required column(s): "
                + ", ".join(name.title() for name in missing)
                + ". Expected headers: Team,Division"
            )

        teams = []
        seen = set()
        for row_number, row in enumerate(reader, start=2):
            name = row[columns["team"]].strip()
            division = row[columns["division"]].strip()
            if not name or not division:
                raise ValueError(f"row {row_number} must include Team and Division")
            if name.casefold() in seen:
                raise ValueError(f"duplicate team name on row {row_number}: {name}")
            seen.add(name.casefold())
            teams.append(TournamentTeam(name, division))

    if not teams:
        raise ValueError("CSV does not contain any teams")
    return teams


def _division_rounds(
    division: str,
    teams: list[TournamentTeam],
    rng: random.Random,
    preferred_bye_team: str | None = None,
    preferred_bye_game: int | None = None,
) -> list[Match]:
    if len(teams) != 5:
        raise ValueError(
            f"division {division!r} has {len(teams)} teams; every division must have exactly 5"
        )

    ordered = teams[:]
    rng.shuffle(ordered)
    if preferred_bye_team is not None:
        if preferred_bye_team not in {team.name for team in ordered}:
            raise ValueError(f"team {preferred_bye_team!r} is not in division {division!r}")
        bye_index = preferred_bye_game - 2
        ordered.remove(next(team for team in ordered if team.name == preferred_bye_team))
        ordered.insert(bye_index, next(team for team in teams if team.name == preferred_bye_team))
    matches = []

    # For five teams, this is the circle-method round robin: one team has a
    # bye, and the other four are paired without repeating an opponent.
    for round_index in range(5):
        bye = ordered[round_index]
        pairings = (
            (ordered[(round_index + 1) % 5], ordered[(round_index + 4) % 5]),
            (ordered[(round_index + 2) % 5], ordered[(round_index + 3) % 5]),
        )
        for match_index, (team_1, team_2) in enumerate(pairings, start=1):
            matches.append(
                Match(
                    game=round_index + 2,
                    match_number=match_index,
                    team_1=team_1.name,
                    team_2=team_2.name,
                    division=division,
                )
            )
        matches.append(
            Match(
                game=round_index + 2,
                match_number=3,
                team_1=bye.name,
                division=division,
            )
        )

    return matches


def _cross_division_pairs(teams: list[TournamentTeam], rng: random.Random) -> list[tuple[TournamentTeam, TournamentTeam]]:
    """Pair all supplied teams across divisions, using backtracking if needed."""
    if len(teams) % 2:
        raise ValueError("an even number of teams is required for exhibition pairings")
    by_division: dict[str, list[TournamentTeam]] = {}
    for team in teams:
        by_division.setdefault(team.division, []).append(team)
    if max(map(len, by_division.values())) > len(teams) // 2:
        raise ValueError("there are not enough teams in other divisions for exhibition pairings")

    remaining = teams[:]
    rng.shuffle(remaining)

    def pair_remaining(unpaired: list[TournamentTeam]) -> list[tuple[TournamentTeam, TournamentTeam]] | None:
        if not unpaired:
            return []
        first = max(
            unpaired,
            key=lambda team: sum(other.division == team.division for other in unpaired),
        )
        candidates = [team for team in unpaired if team.division != first.division]
        rng.shuffle(candidates)
        for second in candidates:
            result = pair_remaining([team for team in unpaired if team not in (first, second)])
            if result is not None:
                return [(first, second)] + result
        return None

    pairs = pair_remaining(remaining)
    if pairs is None:
        raise ValueError("could not create cross-division exhibition pairings")
    return pairs


def create_schedule(
    teams: list[TournamentTeam],
    seed: int | None = None,
    exhibition_bye_team: str | None = None,
    exhibition_bye_game: int | None = None,
) -> list[Match]:
    """Create exhibition game 1 and division games 2 through 6.

    If the number of teams is odd, ``exhibition_bye_team`` identifies the one
    team that skips Game 1. That team can choose its division-play bye with
    ``exhibition_bye_game`` (2 through 6).
    """
    divisions: dict[str, list[TournamentTeam]] = {}
    for team in teams:
        divisions.setdefault(team.division, []).append(team)
    rng = random.Random(seed)
    division_names = sorted(divisions)
    for division in division_names:
        if len(divisions[division]) != 5:
            raise ValueError(
                f"division {division!r} has {len(divisions[division])} teams; every division must have exactly 5"
            )

    team_names = {team.name for team in teams}
    if len(teams) % 2:
        if exhibition_bye_team is None:
            raise ValueError("an exhibition-bye team is required when the total number of teams is odd")
        if exhibition_bye_team not in team_names:
            raise ValueError(f"unknown exhibition-bye team: {exhibition_bye_team}")
    elif exhibition_bye_team is not None:
        raise ValueError("an exhibition-bye team can only be used with an odd number of teams")
    if exhibition_bye_game is not None and exhibition_bye_game not in range(2, 7):
        raise ValueError("exhibition-bye game must be between 2 and 6")
    if exhibition_bye_team is not None and exhibition_bye_game is None:
        raise ValueError("choose the exhibition-bye team's division-play bye with --exhibition-bye-game")

    exhibition_participants = [team for team in teams if team.name != exhibition_bye_team]
    exhibitions = []
    match_number = 1
    for team_1, team_2 in _cross_division_pairs(exhibition_participants, rng):
        exhibitions.append(Match(1, match_number, team_1.name, team_2.name, "", "Exhibition"))
        match_number += 1
    if exhibition_bye_team is not None:
        bye_division = next(team.division for team in teams if team.name == exhibition_bye_team)
        exhibitions.append(Match(1, match_number, exhibition_bye_team, "BYE", bye_division, "Exhibition Bye"))

    division_matches = []
    for division in division_names:
        division_matches.extend(
            _division_rounds(
                division,
                divisions[division],
                rng,
                preferred_bye_team=exhibition_bye_team
                if exhibition_bye_team in {team.name for team in divisions[division]}
                else None,
                preferred_bye_game=exhibition_bye_game,
            )
        )
    return exhibitions + sorted(division_matches, key=lambda match: (match.game, match.division, match.match_number))


def write_schedule(schedule: list[Match], output_file: str | Path) -> None:
    with open(output_file, "w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(["Game", "Match", "Team 1", "Team 2", "Division", "Type"])
        for match in schedule:
            writer.writerow(
                [match.game, match.match_number, match.team_1, match.team_2, match.division, match.match_type]
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a six-game tournament schedule.")
    parser.add_argument("csv_file", type=Path, help="CSV with Team,Division headers")
    parser.add_argument("-o", "--output", type=Path, default=Path("schedule.csv"))
    parser.add_argument("--seed", type=int, help="seed for repeatable pairings")
    parser.add_argument(
        "--exhibition-bye-team",
        help="team that chooses to skip the Game 1 exhibition when the team count is odd",
    )
    parser.add_argument(
        "--exhibition-bye-game",
        type=int,
        choices=range(2, 7),
        metavar="2-6",
        help="division-play game where the exhibition-bye team takes its bye",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    try:
        schedule = create_schedule(
            load_tournament_teams(args.csv_file),
            seed=args.seed,
            exhibition_bye_team=args.exhibition_bye_team,
            exhibition_bye_game=args.exhibition_bye_game,
        )
        write_schedule(schedule, args.output)
    except (OSError, ValueError) as error:
        build_parser().error(str(error))
    print(f"Wrote {len(schedule)} schedule rows to {args.output}")


if __name__ == "__main__":
    main()
