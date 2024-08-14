import random
import math
import pandas as pd
import pulp

class Player:
    def __init__(self, name, rating):
        self.name = name
        self.rating = rating
        
    def pretty_print(self):
        print(f"Player: {self.name}, Rating: {self.rating}")

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
            print(player.name)
        
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
    df = pd.read_csv(file_path)
    players = PlayerPool()
    for _, row in df.iterrows():
        players.add(Player(row['Name'], int(row['Rating'])))
    return players

# Function to calculate the imbalance (objective function)
def calculate_imbalance(teams):
    team_scores = [team.current_score() for team in teams]
    return max(team_scores) - min(team_scores)

# Function to randomly swap players between teams, ensuring constraints are respected
def swap_between_teams(teams, together_constraints, apart_constraints):
    team_a, team_b = random.sample(teams, 2)
    player_a = random.choice(list(team_a.players))
    player_b = random.choice(list(team_b.players))

    # Ensure swapping maintains "together" constraints
    if any(set(c).issubset(team_a.players) for c in together_constraints):
        return
    if any(set(c).issubset(team_b.players) for c in together_constraints):
        return

    # Ensure swapping maintains "apart" constraints
    if team_a.contains_any(apart_constraints.get(player_a, [])) or team_b.contains_any(apart_constraints.get(player_b, [])):
        return

    # Swap players between the two teams
    team_a.swap_players(player_b, player_a)
    team_b.swap_players(player_a, player_b)

# Function to initialize teams, ensuring initial constraints
def initialize_teams(players, num_teams, together_constraints):
    random.shuffle(players)
    teams = []
    assigned = set()

    # Assign together constraints first
    for group in together_constraints:
        group_set = set(group)
        if assigned & group_set:
            continue  # Skip if already assigned
        teams.append(Team(group_set))
        assigned.update(group_set)
    
    # Calculate the size for each team
    total_players = len(players)
    remaining_players = [p for p in players if p not in assigned]
    team_size = total_players // num_teams
    extra_players = total_players % num_teams

    for i in range(num_teams):
        # Start with players already in the team from the together_constraints
        current_team_size = len(teams[i].players) if i < len(teams) else 0
        target_size = team_size + (1 if extra_players > 0 else 0)

        while current_team_size < target_size and remaining_players:
            if i >= len(teams):
                teams.append(Team([]))  # Create new team if needed
            teams[i].players.add(remaining_players.pop())
            current_team_size += 1

        if extra_players > 0:
            extra_players -= 1

    # Ensure exactly num_teams are returned
    if len(teams) > num_teams:
        # Merge smaller teams if needed
        while len(teams) > num_teams:
            smallest_team = min(teams, key=lambda t: len(t.players))
            second_smallest_team = min([t for t in teams if t != smallest_team], key=lambda t: len(t.players))
            second_smallest_team.players.update(smallest_team.players)
            teams.remove(smallest_team)

    return teams

# Simulated Annealing algorithm
def simulated_annealing(players, num_teams=10, together_constraints=[], apart_constraints={}, initial_temp=1000000, cooling_rate=0.00001, min_temp=0.00001):
    # Initialize teams with constraints
    teams = initialize_teams(players, num_teams, together_constraints)
    
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
    
def ilp_team_allocation(player_pool, num_teams, together_constraints, apart_constraints):
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

    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(timeLimit=300))
    
    # Assign players to teams
    teams = [[] for _ in range(num_teams)]
    for player in players:
        for t in range(num_teams):
            if pulp.value(x[player.name, t]) == 1:
                teams[t].append(player)
                break

    return teams

def main():

    # load player and rating
    file_path = 'players.csv'
    players = load_players(file_path)
        
    together_constraints = [
        # [players.look_up_by_name("name 1"), players.look_up_by_name("name 2")]
    ]
    
    apart_constraints = {
        # players.look_up_by_name("name 1"): [players.look_up_by_name("name 2")]
    }

    # Run the Simulated Annealing algorithm
    best_teams, best_imbalance = simulated_annealing(players.get_as_list(), 10, together_constraints, apart_constraints)

    # Print the results
    print("SA Best Teams Configuration:")
    for i, team in enumerate(best_teams, 1):
        print(f"Team {i}:")
        for player in team.players:
            print(player.name)
        print(f"Total Rating: {team.current_score()}")
        print("-" * 20)
    print(f"Best Imbalance: {best_imbalance}")
    
    # Run the ILP team allocation
    num_teams = 10
    teams = ilp_team_allocation(players, num_teams, together_constraints, apart_constraints)
    
    # Print the teams and their total ratings
    for i, team in enumerate(teams, 1):
        print(f"Team {i}:")
        for player in team:
            print(player.name)
        total_rating = sum(player.rating for player in team)
        print(f"Total Rating: {total_rating}")
        print("-" * 20)

if __name__ == "__main__":
    main()
