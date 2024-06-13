
import numpy as np
import random

#Inicializamos primeiramente a tabela com as distências entre as cidades, na matriz "distance":

distances = np.array([
    [0.0, 1.0, 2.2, 2.0, 4.1],
    [1.0, 0.0, 1.4, 2.2, 4.0],
    [2.2, 1.4, 0.0, 2.2, 3.2],
    [2.0, 2.2, 2.2, 0.0, 2.2],
    [4.1, 4.0, 3.2, 2.2, 0.0]
])

#Desenvolvemos um método para o cálculo da distência entre dois pontos (aqui representados como as cidades):


def distance(p1, p2):
    return distances[p1][p2]

#Baseado em um exemplo encontrado em um artigo publicado na internet "Implementing Ant colony optimization in python- solving Traveling salesman problem" (disponível [aqui](https://induraj2020.medium.com/implementation-of-ant-colony-optimization-using-python-solve-traveling-salesman-problem-9c14d3114475)) implementamos o método de otimização para a colônia de formigas. Contudo, realizamos uma alteração, aplicamos a técnica de evaporação de feromônio para garantir que os feromônios não se acumulem indefinidamente e que as soluções mais antigas percam a sua influência ao longo do tempo. Assim, ao inicializar o método ant_colony_optimization, é necessário também passar uma taxa de evaporação (evaporation_rate):

def ant_colony_optimization(n_ants, n_iterations, alpha, beta, evaporation_rate, Q):
    n_points = distances.shape[0]
    pheromone = np.ones((n_points, n_points))
    best_path = None
    best_path_length = np.inf

    for iteration in range(n_iterations):
        paths = []
        path_lengths = []

        for ant in range(n_ants):
            visited = [False] * n_points
            current_point = np.random.randint(n_points)
            visited[current_point] = True
            path = [current_point]
            path_length = 0

            while False in visited:
                unvisited = np.where(np.logical_not(visited))[0]
                probabilities = np.zeros(len(unvisited))

                for i, unvisited_point in enumerate(unvisited):
                    if distance(current_point, unvisited_point) > 0:
                        probabilities[i] = (pheromone[current_point, unvisited_point] ** alpha) * \
                                           ((1 / distance(current_point, unvisited_point)) ** beta)

                probabilities /= np.sum(probabilities)

                next_point = np.random.choice(unvisited, p=probabilities)
                path.append(next_point)
                path_length += distance(current_point, next_point)
                visited[next_point] = True
                current_point = next_point

            paths.append(path)
            path_lengths.append(path_length)

            if path_length < best_path_length:
                best_path = path
                best_path_length = path_length

        # Evaporação do feromônio
        pheromone *= (1 - evaporation_rate)

        # Deposição do feromônio pelas formigas
        for path, path_length in zip(paths, path_lengths):
            for i in range(n_points - 1):
                pheromone[path[i], path[i + 1]] += Q / path_length
            pheromone[path[-1], path[0]] += Q / path_length

    return best_path, best_path_length

#Definimos também o número de formigas da colônia (100 formigas), bem como um total de 100 interações. Também definimos a importÊncia do feromônio (parâmtro alpha), a importância da visibilidade (beta), a taxa de variação do feromônio (rho), a taxa de evaporação do feromônio (evaporation_rate) e a quantidade de feromônio depositado (Q):

n_ants = 10
n_iterations = 100
alpha = 1.0
beta = 2.0
evaporation_rate = 0.5
Q = 100

#Executamos então a ACO, passando os parâmetros:

best_path, best_path_length = ant_colony_optimization(n_ants, n_iterations, alpha, beta, evaporation_rate, Q)

print("Melhor rota:", best_path)
print("Comprimento da melhor rota:", best_path_length)
