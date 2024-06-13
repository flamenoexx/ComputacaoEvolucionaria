import random
import numpy as np

# Definindo os parâmetros do problema
max_IC = 5
max_master = 4
max_phd = 3
max_prof = 2

hours_IC = 160
hours_master = 96
hours_phd = 64
hours_prof = 40

target_hours = 800

# Função Fitness
def fitness(individual):
    total_hours = (individual[0] * hours_IC +
                   individual[1] * hours_master +
                   individual[2] * hours_phd +
                   individual[3] * hours_prof)
    if total_hours > 800:
        return 801
    else:
        return abs(target_hours - total_hours)

# Inicialização da população
def initialize_population(pop_size):
    population = []
    for _ in range(pop_size):
        individual = [random.randint(1, max_IC),
                      random.randint(1, max_master),
                      random.randint(1, max_phd),
                      random.randint(1, max_prof)]
        population.append(individual)
    return population

# Seleção dos pais por torneio
def tournament_selection(population, k=3):
    selected = random.sample(population, k)
    selected.sort(key=fitness)
    return selected[0]

# Crossover
def crossover(parent1, parent2):
    crossover_point = random.randint(1, 3)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Mutação
def mutate(individual, mutation_rate=0.1):
    if random.random() < mutation_rate:
        gene = random.randint(0, 3)
        if gene == 0:
            individual[gene] = random.randint(1, max_IC)
        elif gene == 1:
            individual[gene] = random.randint(1, max_master)
        elif gene == 2:
            individual[gene] = random.randint(1, max_phd)
        elif gene == 3:
            individual[gene] = random.randint(1, max_prof)
    return individual

# Algoritmo Genético
def genetic_algorithm(pop_size=100, generations=100, mutation_rate=0.1, elitism_size=1):
    population = initialize_population(pop_size)
    best_individual = min(population, key=fitness)
    
    for _ in range(generations):
        new_population = []
        
        # Elitismo
        sorted_population = sorted(population, key=fitness)
        new_population.extend(sorted_population[:elitism_size])
        
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, mutation_rate))
            if len(new_population) < pop_size:
                new_population.append(mutate(child2, mutation_rate))
        
        population = new_population
        current_best = min(population, key=fitness)
        if fitness(current_best) < fitness(best_individual):
            best_individual = current_best
    
    total_hours = (best_individual[0] * hours_IC +
                   best_individual[1] * hours_master +
                   best_individual[2] * hours_phd +
                   best_individual[3] * hours_prof)
    return best_individual, fitness(best_individual), total_hours

# Rodando o algoritmo genético 10 vezes
results = [genetic_algorithm() for _ in range(10)]
best_result = min(results, key=lambda x: x[1])
print(f"Melhor solução (AG): {best_result[0]} com aptidão {best_result[1]} e horas {best_result[2]}")