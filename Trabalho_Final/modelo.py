# -*- coding: utf-8 -*-
"""Modelo.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12PZGa9OHvJ1byIjYl7H8RpK6dGNCfW1r
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import lti, step

# Parâmetros do sistema
m = 0.1  # Massa da barra do pêndulo
M = 1.0  # Massa do carrinho
l = 0.5  # Comprimento da barra do pêndulo
b = 0.1  # Coeficiente de atrito do carrinho
d = 0.05  # Coeficiente de atrito do pêndulo
I = 0.006  # Momento de inércia da barra do pêndulo
g = 9.81  # Aceleração da gravidade
K_F = 1.0  # Fator de ganho da força

# Equações do sistema
def pendulum_dynamics(t, y, u):
    x, x_dot, theta, theta_dot = y
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    x_ddot = (K_F * u + m * l * theta_dot**2 * sin_theta - b * x_dot) / (m + M)
    theta_ddot = (m * g * sin_theta - m * l * cos_theta * x_ddot - d * theta_dot) / (I + m * l**2)

    return [x_dot, x_ddot, theta_dot, theta_ddot]

# Controlador PID
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.prev_error = 0

    def control(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

# Simulação
def simulate_pendulum(time_span, initial_conditions, pid):
    t_values = np.linspace(time_span[0], time_span[1], 1000)
    dt = t_values[1] - t_values[0]
    y_values = np.zeros((len(t_values), len(initial_conditions)))
    y_values[0] = initial_conditions

    for i in range(1, len(t_values)):
        t = t_values[i-1]
        y = y_values[i-1]
        error = -y[2]  # Controlar o ângulo theta para zero
        u = pid.control(error, dt)
        sol = solve_ivp(pendulum_dynamics, [t, t + dt], y, args=(u,), t_eval=[t + dt])
        y_values[i] = sol.y.flatten()

    return t_values, y_values

# Parâmetros do controlador
kp = 200
ki = 500
kd = 20
pid = PIDController(kp, ki, kd)

# Condições iniciais: [x, x_dot, theta, theta_dot]
initial_conditions = [0, 0, np.pi / 4, 0]  # Ângulo inicial de 45 graus

# Simulação
time_span = [0, 10]  # Simular por 10 segundos
t_values, y_values = simulate_pendulum(time_span, initial_conditions, pid)

# Plotagem
plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.plot(t_values, y_values[:, 0], label='Posição do Carrinho')
plt.ylabel('Posição (m)')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t_values, y_values[:, 2], label='Ângulo do Pêndulo')
plt.ylabel('Ângulo (rad)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t_values, y_values[:, 1], label='Velocidade do Carrinho')
plt.plot(t_values, y_values[:, 3], label='Velocidade Angular do Pêndulo')
plt.ylabel('Velocidade (m/s e rad/s)')
plt.xlabel('Tempo (s)')
plt.legend()

plt.tight_layout()
plt.show()