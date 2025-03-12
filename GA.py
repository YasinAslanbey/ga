import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# File path
file_path1 = 'C:/Users/Asus/OneDrive/Documents/Python/Berlin52.tsp'

# Parser and sorting ------------------------------------------------------------
def unclear_data_reader(fp):
    """
    Reads the file, extracts numeric lines, and converts them into a DataFrame
    with columns 'num', 'X', and 'Y'.
    """
    with open(fp, 'r') as file:
        lines = file.readlines()

    clean_rows = [line.strip() for line in lines if line.strip() and line.strip()[0].isdigit()]
    data = [row.split() for row in clean_rows]
    clean_df = pd.DataFrame(data, columns=['num', 'X', 'Y'])

    clean_df['num'] = clean_df['num'].astype(int)
    clean_df['X'] = clean_df['X'].astype(float)
    clean_df['Y'] = clean_df['Y'].astype(float)

    return clean_df

df1 = unclear_data_reader(file_path1)
print("First file preview:\n", df1.head())

# Distance calculator -----------------------------------------------------------
def calculate_distance(city1, city2):
    """
    Calculates the Euclidean distance between two cities
    based on their 'X' and 'Y' coordinates.
    """
    x1, y1 = city1['X'], city1['Y']
    x2, y2 = city2['X'], city2['Y']
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Random solution ---------------------------------------------------------------
def generate_random_solution(df):
    """
    Generates a random sequence of city numbers as a potential solution.
    """
    Random_list = list(df['num'])
    random.shuffle(Random_list)
    return Random_list

# Fitness -----------------------------------------------------------------------
def calculate_fitness(Random_list, city_data):
    """
    Computes the total distance for a given route,
    including the return to the starting city.
    """
    total_distance = 0.0
    num_cities = len(Random_list)

    for i in range(num_cities - 1):
        city1 = city_data.loc[city_data['num'] == Random_list[i]].iloc[0]
        city2 = city_data.loc[city_data['num'] == Random_list[i + 1]].iloc[0]
        total_distance += calculate_distance(city1, city2)

    city_start = city_data.loc[city_data['num'] == Random_list[0]].iloc[0]
    city_end = city_data.loc[city_data['num'] == Random_list[-1]].iloc[0]
    total_distance += calculate_distance(city_start, city_end)

    return total_distance

# Greedy algorithm -------------------------------------------------------------
def greedy_solution(df, start_city):
    """
    Constructs a route starting from a city by always selecting the nearest unvisited city.
    """
    unvisited = df.copy()
    visited = []
    current_city = start_city

    while not unvisited.empty:
        visited.append(current_city)
        unvisited = unvisited[unvisited['num'] != current_city]

        if not unvisited.empty:
            current_city_coords = df.loc[df['num'] == current_city, ['X', 'Y']].iloc[0]
            distances = ((unvisited[['X', 'Y']] - current_city_coords) ** 2).sum(axis=1)
            next_city_idx = distances.idxmin()

            if next_city_idx in unvisited.index:
                next_city = unvisited.loc[next_city_idx, 'num']
                current_city = next_city
            else:
                break

    return visited

# Initial population -----------------------------------------------------------
def initialize_population(df, population_size, greedy_fraction=0.1):
    """
    Creates an initial population with a mix of greedy and random solutions.
    """
    population = []
    n_greedy = int(population_size * greedy_fraction)

    for _ in range(n_greedy):
        start_city = random.choice(df['num'])
        solution = greedy_solution(df, start_city)
        population.append(solution)

    for _ in range(population_size - n_greedy):
        solution = generate_random_solution(df)
        population.append(solution)

    return population

# Selection ---------------------------------------------------------------------
def tournament_selection(population, fitnesses, tournament_size=3):
    """
    Selects the best solution from a random subset of the population.
    """
    participants = random.sample(list(zip(population, fitnesses)), tournament_size)
    winner = min(participants, key=lambda x: x[1])
    return winner[0]

# Crossover ---------------------------------------------------------------------
def pmx_crossover(parent1, parent2):
    """
    Partially matched crossover (PMX) combines two parent solutions
    to create a new child solution.
    """
    size = len(parent1)
    child = [None] * size
    pt1, pt2 = sorted(random.sample(range(size), 2))

    child[pt1:pt2+1] = parent1[pt1:pt2+1]

    for gene in parent2:
        if gene not in child:
            for i in range(size):
                if child[i] is None:
                    child[i] = gene
                    break

    return child

# Mutation ----------------------------------------------------------------------
def swap_mutation(solution, mutation_rate=0.01):
    """
    Mutates a solution by swapping two cities with a certain probability.
    """
    for i in range(len(solution)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(solution) - 1)
            solution[i], solution[j] = solution[j], solution[i]
    return solution

# Epoch creation ---------------------------------------------------------------
def create_new_population(df, population, mutation_rate):
    """
    Creates the next generation of solutions using selection, crossover, and mutation.
    """
    fitnesses = [calculate_fitness(solution, df) for solution in population]
    new_population = []

    for _ in range(len(population)):
        parent1 = tournament_selection(population, fitnesses)
        parent2 = tournament_selection(population, fitnesses)
        child = pmx_crossover(parent1, parent2)
        child = swap_mutation(child, mutation_rate)
        new_population.append(child)

    return new_population

# Run genetic algorithm --------------------------------------------------------
def run_genetic_algorithm(df, population_size, num_generations, mutation_rate):
    """
    Executes the genetic algorithm and finds the best solution.
    """
    population = initialize_population(df, population_size)
    best_fitness = float('inf')
    best_solution = None
    all_best_fitnesses = []  # Track best fitness at each generation

    for generation in range(num_generations):
        population = create_new_population(df, population, mutation_rate)
        fitnesses = [calculate_fitness(solution, df) for solution in population]
        generation_best = min(fitnesses)
        all_best_fitnesses.append(generation_best)  # Store best fitness of each generation

        if generation_best < best_fitness:
            best_fitness = generation_best
            best_solution = population[fitnesses.index(generation_best)]

        print(f"Generation {generation+1}: Best Fitness = {generation_best}")

    return best_solution, best_fitness, all_best_fitnesses

# Task 5 - Visualizing the Best Route -----------------------------------------
def visualize_route(df, best_solution):
    """
    Visualizes the best route found by the genetic algorithm.
    """
    # Extract coordinates for the best solution
    best_cities = df[df['num'].isin(best_solution)]
    route_coords = best_cities[['X', 'Y']].values

    # Add the starting city to the end of the route to close the loop
    route_coords = np.vstack([route_coords, route_coords[0]])

    # Plot the cities and the route
    plt.figure(figsize=(10, 6))
    plt.plot(route_coords[:, 0], route_coords[:, 1], 'b-o', markersize=6)
    plt.scatter(df['X'], df['Y'], color='red', marker='x')  # cities as red 'x'
    
    for i, city in df.iterrows():
        plt.text(city['X'], city['Y'], str(city['num']), fontsize=9)

    plt.title('Best Route Found by Genetic Algorithm')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

# Run the algorithm
best_solution, best_fitness, all_best_fitnesses = run_genetic_algorithm(df1, population_size=100, num_generations=50, mutation_rate=0.01)
print("Best solution:", best_solution)
print("Best fitness:", best_fitness)

# Visualize the best route
visualize_route(df1, best_solution)

# Plot fitness over generations
plt.figure(figsize=(10, 6))
plt.plot(all_best_fitnesses, 'g-', label='Best Fitness')
plt.title('Best Fitness Over Generations')
plt.xlabel('Generation')
plt.ylabel('Best Fitness (lower is better)')
plt.legend()
plt.show()
