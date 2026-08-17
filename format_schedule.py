"""Print a tournament schedule in an easy-to-read round-by-round format."""

import argparse
import csv
from pathlib import Path

from tournament import Match


def load_schedule(csv_file: str | Path) -> list[Match]:
    with open(csv_file, newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"Game", "Match", "Team 1", "Team 2", "Division", "Type"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("schedule CSV must have Game, Match, Team 1, Team 2, Division, and Type headers")
        return [
            Match(
                game=int(row["Game"]),
                match_number=int(row["Match"]),
                team_1=row["Team 1"],
                team_2=row["Team 2"],
                division=row["Division"],
                match_type=row["Type"],
            )
            for row in reader
        ]


def format_schedule(schedule) -> str:
    lines = []
    for game in sorted({match.game for match in schedule}):
        title = "ROUND 1 (EXHIBITION)" if game == 1 else f"ROUND {game} (GAME {game - 1} of 4)"
        lines.extend([title, "---------"])
        for match in (match for match in schedule if match.game == game):
            lines.append(f"{match.division or 'EXHIBITION'} - {match.team_1} - {match.team_2}")
        lines.append("")
    return "\n".join(lines).rstrip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print a readable tournament schedule.")
    parser.add_argument("csv_file", type=Path, help="generated schedule CSV")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    try:
        print(format_schedule(load_schedule(args.csv_file)))
    except (OSError, ValueError) as error:
        build_parser().error(str(error))


if __name__ == "__main__":
    main()
