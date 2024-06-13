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
def fitness(position):
    total_hours = (position[0] * hours_IC +
                   position[1] * hours_master +
                   position[2] * hours_phd +
                   position[3] * hours_prof)
    if total_hours > 800:
        return 801
    else:
        return abs(target_hours - total_hours)

# Parâmetros PSO 
num_particles = 30
num_iterations = 100
w = 0.5
c1 = 1.5
c2 = 1.5

class Particle:
    def __init__(self):
        self.position = np.array([random.randint(1, max_IC),
                                  random.randint(1, max_master),
                                  random.randint(1, max_phd),
                                  random.randint(1, max_prof)])
        self.velocity = np.random.uniform(-1, 1, 4)
        self.best_position = self.position.copy()
        self.best_fitness = fitness(self.position)
    
    def update_velocity(self, global_best_position):
        r1 = np.random.uniform(size=4)
        r2 = np.random.uniform(size=4)
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive + social
    
    def update_position(self):
        self.position = np.clip(self.position + self.velocity, [1, 1, 1, 1], [max_IC, max_master, max_phd, max_prof])
        current_fitness = fitness(self.position)
        if current_fitness < self.best_fitness:
            self.best_position = self.position.copy()
            self.best_fitness = current_fitness

def pso():
    particles = [Particle() for _ in range(num_particles)]
    global_best_position = min(particles, key=lambda p: p.best_fitness).best_position
    
    for _ in range(num_iterations):
        for particle in particles:
            particle.update_velocity(global_best_position)
            particle.update_position()
        
        global_best_position = min(particles, key=lambda p: p.best_fitness).best_position
    
    best_particle = min(particles, key=lambda p: p.best_fitness)
    total_hours = (best_particle.best_position[0] * hours_IC +
                   best_particle.best_position[1] * hours_master +
                   best_particle.best_position[2] * hours_phd +
                   best_particle.best_position[3] * hours_prof)
    return np.round(best_particle.best_position).astype(int), best_particle.best_fitness, total_hours

# Rodando o PSO 10 vezes
results = [pso() for _ in range(10)]
best_result = min(results, key=lambda x: x[1])
print(f"Melhor solução (PSO): {best_result[0]} com aptidão {best_result[1]} e horas {best_result[2]}")