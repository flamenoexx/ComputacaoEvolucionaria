import numpy as np
import matplotlib.pyplot as plt
import time

# Função objetivo
def objective_function(x):
    return x**3

# Implementação do PSO para maximização
def pso_maximize(objective_func, num_particles, max_iter, bounds):
    # Parâmetros do PSO
    w = 0.5    # inércia
    c1 = 1.5   # constante cognitiva
    c2 = 1.5   # constante social

    # Inicialização
    particle_position = np.random.uniform(bounds[0], bounds[1], (num_particles, 1))
    particle_velocity = np.zeros((num_particles, 1))
    pbest_position = particle_position.copy()
    pbest_value = objective_func(particle_position)
    gbest_idx = np.argmax(pbest_value)
    gbest_value = pbest_value[gbest_idx]
    gbest_position = pbest_position[gbest_idx]

    # Listas para armazenar os resultados ao longo das iterações
    iteration_best_values = []
    iteration_gbest_values = []
    particle_positions_over_iterations = []  # Armazenar posições das partículas ao longo das iterações

    # Tempo de processamento
    start_time = time.time()

    # Execução do PSO
    for i in range(max_iter):
        # Armazenar posições atuais das partículas
        particle_positions_over_iterations.append(particle_position.copy())

        # Atualização das partículas
        r1 = np.random.rand(num_particles, 1)
        r2 = np.random.rand(num_particles, 1)

        particle_velocity = w * particle_velocity \
                             + c1 * r1 * (pbest_position - particle_position) \
                             + c2 * r2 * (gbest_position - particle_position)

        particle_position = particle_position + particle_velocity

        # Garantir que as posições das partículas estejam dentro dos limites
        particle_position = np.clip(particle_position, bounds[0], bounds[1])

        # Avaliação do fitness
        fitness_values = objective_func(particle_position)

        # Atualização pbest
        update_idx = fitness_values > pbest_value
        pbest_position[update_idx] = particle_position[update_idx]
        pbest_value[update_idx] = fitness_values[update_idx]

        # Atualização gbest
        gbest_idx = np.argmax(pbest_value)
        gbest_value = pbest_value[gbest_idx]
        gbest_position = pbest_position[gbest_idx]

        # Armazenar os melhores valores encontrados em cada iteração
        iteration_best_values.append(np.max(pbest_value))
        iteration_gbest_values.append(gbest_value)

    # Tempo de processamento
    end_time = time.time()
    processing_time = end_time - start_time

    # Retornar os valores de gbest ao longo das iterações, o tempo de processamento e as posições das partículas
    return iteration_gbest_values, processing_time, particle_positions_over_iterations

# Parâmetros do PSO
num_particles = 30
max_iter = 100
bounds = [0, 35]

# Executar PSO para maximização
iteration_gbest_values, processing_time, particle_positions_over_iterations = pso_maximize(objective_function, num_particles, max_iter, bounds)

# Plotar os resultados
plt.figure(figsize=(10, 6))
plt.plot(range(max_iter), iteration_gbest_values, marker='o', color='b', linestyle='-', linewidth=1, markersize=4)
plt.xlabel('Iteração')
plt.ylabel('Valor de $x^3$')
plt.title('Convergência do PSO para Maximização de $x^3$')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plotar as posições das partículas ao longo das iterações
plt.figure(figsize=(10, 6))
for i in range(num_particles):
    positions = [position[i, 0] for position in particle_positions_over_iterations]
    plt.plot(range(max_iter), positions, label=f'Partícula {i+1}')
plt.xlabel('Iteração')
plt.ylabel('Posição')
plt.title('Posição das Partículas ao Longo das Iterações')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()

# Exibir tempo de processamento
print(f"Tempo de processamento: {processing_time:.4f} segundos")