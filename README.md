# gilman_classic
Team building algorithm for the Gilman Classic charity vb tournament

I've put in both simulated annealing and also inductive logic programming. going to use the ILP results because it's guaranteed to be more optimal when presented with constraints. 

unfortunately, the algorithm will take forever to find a truly optimal solution, but you can set a block on how long to run in `prob.solve(pulp.PULP_CBC_CMD(timeLimit=300))`

i won't be releasing the player ratings so don't even try me on that

## Usage

Create a project-local virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If `pip` reports a `401` from an AWS CodeArtifact index, your global `pip`
configuration is taking precedence. Install from public PyPI explicitly:

```bash
python -m pip install --index-url https://pypi.org/simple -r requirements.txt
```

When you return to the project in a new terminal, activate it again with
`source .venv/bin/activate`.

The player file must begin with a header row containing `Name`, `Rating`, and `Phone Number` columns:

```csv
Name,Rating,Phone Number
Alex,8,555-0100
Jordan,6,555-0101
```

Run the default workflow (both algorithms, 10 teams):

```bash
python sorting.py players.csv
```

To see all options:

```bash
python sorting.py -h
```

Useful examples:

```bash
python sorting.py players.csv --teams 8 --algorithm ilp --time-limit 600
python sorting.py players.csv --algorithm annealing --seed 42
python sorting.py players.csv --constraints constraints.json
```

The simulated-annealing defaults use a starting temperature of `100`, a
cooling rate of `0.9995`, and a minimum temperature of `0.01`, which gives it
roughly 18,000 swap attempts. The ILP solver runs for 300 seconds by default,
so `--algorithm ilp` already matches a five-minute run.

Together/apart constraints can be kept in a separate JSON file:

```json
{
  "together": [
    ["Alex", "Jordan"]
  ],
  "apart": [
    ["Alex", "Taylor"],
    ["Jordan", "Morgan"]
  ]
}
```

Run with:

```bash
python sorting.py players.csv --constraints constraints.json
```

Names must match players in the CSV, but capitalization and extra spaces do
not matter. Each `apart` pair is treated as mutual, so only one direction is
needed.

## Tournament schedule

Create a `Team,Division` CSV with five teams in each division. The scheduler
creates game 1 as cross-division exhibition play, then games 2 through 6 as
division play. Each team plays each other team in its division once and has
one bye. If the total number of teams is odd, specify the team that chooses
the Game 1 exhibition bye and the division-play game where it wants its bye.

```csv
Team,Division
A1,North
A2,North
A3,North
A4,North
A5,North
B1,South
B2,South
B3,South
B4,South
B5,South
```

Generate `schedule.csv` with:

```bash
python tournament.py teams.csv --output schedule.csv --seed 42
```

For three divisions, for example:

```bash
python tournament.py teams.csv --exhibition-bye-team A1 --exhibition-bye-game 6
```

To print the generated schedule in a readable round-by-round format:

```bash
python format_schedule.py schedule.csv
```
