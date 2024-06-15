import numpy as np

class Particle:
    def __init__(self, bounds, dimensions):
        self.position = np.random.uniform(bounds[0], bounds[1], dimensions)
        self.velocity = np.random.uniform(-1, 1, dimensions)
        self.best_position = self.position.copy()
        self.best_value = float('inf')

    def update_velocity(self, global_best_position, w, c1, c2):
        r1 = np.random.random(self.velocity.shape)
        r2 = np.random.random(self.velocity.shape)
        cognitive_component = c1 * r1 * (self.best_position - self.position)
        social_component = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive_component + social_component

    def update_position(self, bounds):
        self.position += self.velocity
        self.position = np.clip(self.position, bounds[0], bounds[1])

def objective_function(x):
    return np.sum(x**2)

def pso(num_particles, dimensions, bounds, max_iterations, w_min, w_max, c1, c2):
    particles = [Particle(bounds, dimensions) for _ in range(num_particles)]
    global_best_position = particles[0].position.copy()
    global_best_value = float('inf')

    # Linear weight decay
    w = np.linspace(w_max, w_min, max_iterations)

    for iteration in range(max_iterations):
        for particle in particles:
            value = objective_function(particle.position)
            if value < particle.best_value:
                particle.best_value = value
                particle.best_position = particle.position.copy()
            if value < global_best_value:
                global_best_value = value
                global_best_position = particle.position.copy()

        for particle in particles:
            particle.update_velocity(global_best_position, w[iteration], c1, c2)
            particle.update_position(bounds)

        if iteration % 100 == 0:
            print(f"Iteration: {iteration}, Global Best Value: {global_best_value}")

    return global_best_position, global_best_value

# Parameters
num_particles = 50
dimensions = 5
bounds = (-10, 10)
max_iterations = 1000
w_min = 0.1
w_max = 1.1
c1 = 1
c2 = 1

# Run PSO
best_position, best_value = pso(num_particles, dimensions, bounds, max_iterations, w_min, w_max, c1, c2)
print(f"Best Position: {best_position}, Best Value: {best_value}")