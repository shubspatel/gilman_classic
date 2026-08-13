import argparse
import json
import random
import math
from pathlib import Path

class Player:
    def __init__(self, name, rating, phone_number):
        self.name = name
        self.rating = rating
        self.phone_number = phone_number
        
    def pretty_print(self):
        print(f"Player: {self.name}, Rating: {self.rating}, Phone: {self.phone_number}")

class Team:
    def __init__(self, players):
        self.players = set(players)
    
    def current_score(self):
        return sum(player.rating for player in self.players)

    def swap_players(self, new, old):
        if old in self.players:
            self.players.remove(old)  # Remove the old player
            self.players.add(new)     # Add the new player

    def contains_any(self, players):
        return any(player in self.players for player in players)
        
    def pretty_print(self):
        print(f"Team!")
        for player in self.players:
            print(f"{player.name} ({player.phone_number})")
        
class PlayerPool:
    def __init__(self):
        self.players = set()
        
    def add(self, player):
        self.players.add(player)
        
    def look_up_by_name(self, name):
        for player in self.players:
            if player.name.lower() == name.lower():
                return player
        return None  # Return None if no player is found
        
    def get_as_list(self):
        return list(self.players)

# Function to print all of the teams
def print_teams(teams):
    for team in teams:
        team.pretty_print()
        print("------")
    print("-----------")
    
def load_players(file_path):
    import pandas as pd

    required_columns = ("name", "rating", "phone number")
    df = pd.read_csv(file_path, dtype=str)
    columns = {
        " ".join(str(column).strip().lower().split()): column
        for column in df.columns
    }
    missing_columns = [column for column in required_columns if column not in columns]
    if missing_columns:
        expected = ", ".join(required_columns)
        missing = ", ".join(missing_columns)
        raise ValueError(f"CSV is missing required column(s): {missing}. Expected: {expected}")

    players = PlayerPool()
    for _, row in df.iterrows():
        players.add(
            Player(
                row[columns["name"]],
                int(row[columns["rating"]]),
                row[columns["phone number"]],
            )
        )
    return players


def load_constraints(file_path, players):
    """Load together/apart constraints from a JSON file using player names."""
    if file_path is None:
        return [], {}

    with open(file_path, encoding="utf-8") as constraints_file:
        config = json.load(constraints_file)

    def find_player(name):
        if not isinstance(name, str):
            raise ValueError(f"player names must be strings; got {name!r}")
        player = players.look_up_by_name(name)
        if player is None:
            raise ValueError(f"constraint refers to unknown player: {name}")
        return player

    together_constraints = []
    for group in config.get("together", []):
        if not isinstance(group, list) or len(group) < 2:
            raise ValueError("each 'together' entry must be a list of at least 2 names")
        together_constraints.append([find_player(name) for name in group])

    apart_constraints = {}
    for pair in config.get("apart", []):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("each 'apart' entry must be a list of exactly 2 names")
        player_a, player_b = (find_player(name) for name in pair)
        apart_constraints.setdefault(player_a, []).append(player_b)
        apart_constraints.setdefault(player_b, []).append(player_a)

    return together_constraints, apart_constraints

# Function to calculate the imbalance (objective function)
def calculate_imbalance(teams):
    team_scores = [team.current_score() for team in teams]
    return max(team_scores) - min(team_scores)

# Function to randomly swap players between teams, ensuring constraints are respected
def has_apart_conflict(players, apart_constraints):
    player_set = set(players)
    return any(
        other in player_set
        for player in player_set
        for other in apart_constraints.get(player, [])
    )


def swap_between_teams(teams, together_constraints, apart_constraints):
    team_a, team_b = random.sample(teams, 2)
    player_a = random.choice(list(team_a.players))
    player_b = random.choice(list(team_b.players))

    # Ensure swapping maintains "together" constraints
    if any(set(c).issubset(team_a.players) for c in together_constraints):
        return
    if any(set(c).issubset(team_b.players) for c in together_constraints):
        return

    # Check the teams after the proposed swap. This both preserves valid
    # apart constraints and allows an invalid initial assignment to be fixed.
    candidate_a = (team_a.players - {player_a}) | {player_b}
    candidate_b = (team_b.players - {player_b}) | {player_a}
    if has_apart_conflict(candidate_a, apart_constraints) or has_apart_conflict(candidate_b, apart_constraints):
        return

    # Swap players between the two teams
    team_a.swap_players(player_b, player_a)
    team_b.swap_players(player_a, player_b)

# Function to initialize teams, ensuring initial constraints
def initialize_teams(players, num_teams, together_constraints, apart_constraints):
    random.shuffle(players)
    teams = [Team([]) for _ in range(num_teams)]
    assigned = set()

    # Treat together groups as units when creating the initial assignment.
    groups = []
    for group in together_constraints:
        group_set = set(group)
        if assigned & group_set:
            continue  # Skip if already assigned
        groups.append(group_set)
        assigned.update(group_set)

    total_players = len(players)
    remaining_players = [p for p in players if p not in assigned]
    team_size = total_players // num_teams
    extra_players = total_players % num_teams
    target_sizes = [team_size + (i < extra_players) for i in range(num_teams)]

    def choose_team(player_group):
        valid_teams = [
            i for i, team in enumerate(teams)
            if len(team.players) + len(player_group) <= target_sizes[i]
            and not has_apart_conflict(team.players | set(player_group), apart_constraints)
        ]
        if not valid_teams:
            valid_teams = [
                i for i, team in enumerate(teams)
                if len(team.players) + len(player_group) <= target_sizes[i]
            ]
        if not valid_teams:
            valid_teams = list(range(num_teams))
        return min(valid_teams, key=lambda i: (len(teams[i].players), teams[i].current_score()))

    for group in sorted(groups, key=len, reverse=True):
        teams[choose_team(group)].players.update(group)

    for player in remaining_players:
        teams[choose_team({player})].players.add(player)

    return teams

# Simulated Annealing algorithm
def simulated_annealing(
    players,
    num_teams=10,
    together_constraints=[],
    apart_constraints={},
    initial_temp=100,
    cooling_rate=0.9995,
    min_temp=0.01,
):
    # Initialize teams with constraints
    teams = initialize_teams(players, num_teams, together_constraints, apart_constraints)
    
    current_imbalance = calculate_imbalance(teams)
    best_teams = [Team(set(team.players)) for team in teams]  # Deep copy of teams
    best_imbalance = current_imbalance
    temp = initial_temp
    
    while temp > min_temp:
        # Create a new candidate solution by swapping players between teams
        new_teams = [Team(set(team.players)) for team in teams]  # Copy current teams
        swap_between_teams(new_teams, together_constraints, apart_constraints)
        
        # Calculate the new imbalance
        new_imbalance = calculate_imbalance(new_teams)
        
        # Decide whether to accept the new solution
        if new_imbalance < current_imbalance or random.random() < math.exp((current_imbalance - new_imbalance) / temp):
            teams = new_teams
            current_imbalance = new_imbalance
            
            # Update the best solution found so far
            if current_imbalance < best_imbalance:
                best_teams = [Team(set(team.players)) for team in teams]
                best_imbalance = current_imbalance
        
        # Cool down the temperature
        temp *= cooling_rate
    
    return best_teams, best_imbalance
    
def ilp_team_allocation(player_pool, num_teams, together_constraints, apart_constraints, time_limit=300):
    import pulp

    players = player_pool.get_as_list()
    
    # Define the problem
    prob = pulp.LpProblem("TeamAssignment", pulp.LpMinimize)

    # Create variables
    x = pulp.LpVariable.dicts("PlayerTeam",
                              ((player.name, team) for player in players for team in range(num_teams)),
                              cat='Binary')


    # Variables for maximum and minimum team scores to minimize the difference
    max_score = pulp.LpVariable("max_score", lowBound=0)
    min_score = pulp.LpVariable("min_score", lowBound=0)

    # Objective function: Minimize the difference between the maximum and minimum team scores
    prob += max_score - min_score

    # Calculate team scores and relate them to max and min scores
    team_scores = [pulp.lpSum(x[player.name, t] * player.rating for player in players) for t in range(num_teams)]
    for t in range(num_teams):
        prob += team_scores[t] <= max_score
        prob += team_scores[t] >= min_score

    # Constraint: Every player is assigned to exactly one team
    for player in players:
        prob += pulp.lpSum(x[player.name, t] for t in range(num_teams)) == 1

    # Constraint: Each team has approximately the same number of players
    total_players = len(players)
    min_team_size = total_players // num_teams
    max_team_size = min_team_size + (1 if total_players % num_teams != 0 else 0)
    
    for t in range(num_teams):
        prob += pulp.lpSum(x[player.name, t] for player in players) >= min_team_size
        prob += pulp.lpSum(x[player.name, t] for player in players) <= max_team_size

    # Together constraints
    for group in together_constraints:
        for t in range(num_teams):
            prob += sum(x[player.name, t] for player in group) == len(group) * x[next(iter(group)).name, t]

    # Apart constraints
    for apart_group in apart_constraints.items():
        player1, players_not_with = apart_group
        for player2 in players_not_with:
            for t in range(num_teams):
                prob += x[player1.name, t] + x[player2.name, t] <= 1

    # Solve the problem quietly; provide a concise progress indication instead
    # of printing CBC's full model and branch-and-bound log.
    print(f"Running ILP solver (up to {time_limit} seconds)...", flush=True)
    prob.solve(pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=False))
    print(f"ILP solver finished: {pulp.LpStatus[prob.status]}", flush=True)
    
    # Assign players to teams
    teams = [[] for _ in range(num_teams)]
    for player in players:
        for t in range(num_teams):
            if pulp.value(x[player.name, t]) == 1:
                teams[t].append(player)
                break

    return teams

def build_parser():
    parser = argparse.ArgumentParser(
        description="Build balanced volleyball teams from a player ratings CSV.",
        epilog=(
        "CSV format: the first row must include 'Name', 'Rating', and "
            "'Phone Number'. Example:\n"
            "  Name,Rating,Phone Number\n"
            "  Alex,8,555-0100\n"
            "  Jordan,6,555-0101\n\n"
            "Together/apart constraints can be supplied with --constraints."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        default="players.csv",
        help="path to the player ratings CSV (default: players.csv)",
    )
    parser.add_argument(
        "-n",
        "--teams",
        type=int,
        default=10,
        metavar="N",
        help="number of teams to create (default: 10)",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        choices=("both", "ilp", "annealing"),
        default="both",
        help="which allocation method to run (default: both)",
    )
    parser.add_argument(
        "-t",
        "--time-limit",
        type=int,
        default=300,
        metavar="SECONDS",
        help="maximum ILP solver time in seconds (default: 300)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed for repeatable simulated-annealing results",
    )
    parser.add_argument(
        "-c",
        "--constraints",
        type=Path,
        metavar="JSON",
        help="JSON file containing together/apart player constraints",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.teams < 1:
        build_parser().error("--teams must be at least 1")
    if args.time_limit < 1:
        build_parser().error("--time-limit must be at least 1 second")

    csv_path = Path(args.csv_file)
    if not csv_path.is_file():
        build_parser().error(f"player CSV not found: {csv_path}")
    if args.constraints is not None and not args.constraints.is_file():
        build_parser().error(f"constraints file not found: {args.constraints}")

    if args.seed is not None:
        random.seed(args.seed)

    # load player and rating
    players = load_players(csv_path)
        
    try:
        together_constraints, apart_constraints = load_constraints(args.constraints, players)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        build_parser().error(f"invalid constraints file: {error}")

    if args.algorithm in ("both", "annealing"):
        best_teams, best_imbalance = simulated_annealing(
            players.get_as_list(), args.teams, together_constraints, apart_constraints
        )

        print("SA Best Teams Configuration:")
        for i, team in enumerate(best_teams, 1):
            print(f"Team {i}:")
            for player in team.players:
                print(f"{player.name} ({player.phone_number})")
            print(f"Total Rating: {team.current_score()}")
            print("-" * 20)
        print(f"Best Imbalance: {best_imbalance}")

    if args.algorithm in ("both", "ilp"):
        teams = ilp_team_allocation(
            players, args.teams, together_constraints, apart_constraints, args.time_limit
        )

        print("ILP Team Configuration:")
        for i, team in enumerate(teams, 1):
            print(f"Team {i}:")
            for player in team:
                print(f"{player.name} ({player.phone_number})")
            total_rating = sum(player.rating for player in team)
            print(f"Total Rating: {total_rating}")
            print("-" * 20)

if __name__ == "__main__":
    main()
