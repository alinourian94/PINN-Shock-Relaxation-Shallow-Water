# PINN Implementation for Variable Bed Slope Supercritical Flow Problem

import numpy as np
import sciann as sn
import matplotlib.pyplot as plt
from sciann.utils.math import diff, exp, abs
from scipy.optimize import fsolve

# Variables and Functions
x = sn.Variable("x")
t = sn.Variable("t")         

y = sn.Functional("y", [t, x], 6 * [60], "tanh")
V = sn.Functional("V", [t, x], 6 * [60], "tanh")

g = 9.81 
# Bed slope defined only in the domain 8 < x < 12 using step functions
S = 0.2 - (0.05 * (x - 10.0)**2)
slope_activation = sn.step(x, 7.99) - sn.step(x, 12.0)  # Active only between 8 and 12
Z = y - slope_activation * S   

# Governing Equations 
L1 = diff(V * y, x) + diff(y, t) # Continuity Equation
L2 = (1/g) * (diff(V, t) + V * diff(V, x)) + diff(Z, x) # Momentum Equation

# Initial and Boundary Conditions using smooth step
def sigmoid(x, beta=50):
    return 1 / (1 + exp(-beta * x))

ic_y = sigmoid(t, beta=50) * abs(Z - 2.0)    
ic_V = sigmoid(t, beta=50) * abs(V)          

bc_V_left = (x == 0.0) * abs(V - 12.52835)
bc_Z_left = (x == 0.0) * abs(Z - 2.0)

# Learning Rate Definition
initial_lr = 1e-3
final_lr = initial_lr/10
EPOCHS=50
learning_rate_finall = {
    "scheduler": "ExponentialDecay", 
    "initial_learning_rate": initial_lr,
    "final_learning_rate": final_lr, 
    "decay_epochs": EPOCHS
}

# SciANN Model
model = sn.SciModel(
    [t, x],
    [L1, L2, ic_y, ic_V, bc_V_left, bc_Z_left],
    "mse",
    optimizer="Adam"
)

#  Training Data
x_data = np.concatenate([
    np.linspace(0, 7.8, 40),
    np.linspace(7.8, 12.5, 80),
    np.linspace(12.5, 25, 40)
])
t_data = np.linspace(0., 200, 200)

t_grid, x_grid = np.meshgrid(t_data, x_data)

# Training Model
history = model.train(
    [t_grid, x_grid],
    6 * ['zero'],
    epochs = 500,
    batch_size = 200,
    learning_rate = learning_rate_finall,
    adaptive_weights={'method': 'NTK', 'freq': 25, 'use_score': True}
)

# Evaluation
t_test, x_test = np.meshgrid(200, np.linspace(0, 25, 100))
y_pred = y.eval(model, [t_test, x_test])
V_pred = V.eval(model, [t_test, x_test])
Z_pred = Z.eval(model, [t_test, x_test])

# Exact Solution 
g = 9.81
y1 = 2
q = 25.0567
V1 = q / y1

x_exact = np.linspace(0, 25, 100)

dd_exact = -1 * (np.heaviside(x_exact < 8, 0) - np.heaviside(x_exact < 12, 0))
Z_exact = dd_exact * (0.2 - 0.05 * (x_exact - 10)**2)

def energy_eq(y, E):
    return y + (q**2 / (2 * g * y**2)) - E

y_values = []
for z in Z_exact:
    E = y1 + (V1**2 / (2 * g)) - z
    result = fsolve(energy_eq, y1, args=(E,))
    y_values.append(result[0])

h_exact = np.array(y_values) + Z_exact

# Error Metrics
yy_pred = y_pred.reshape(100, 1).flatten()
mse = np.mean((h_exact - yy_pred)**2)
rmsd = np.sqrt(mse)
print("MSE:", mse)
print("RMSD:", rmsd)

# Plot Results
fig, _ = plt.subplots(1, 3, figsize=(20, 5))

plt.subplot(1, 3, 1)
plt.plot(x_exact, yy_pred, color="blue", label='PINN', linewidth=2)
plt.plot(x_exact, h_exact, linestyle='--', color='red', label='Exact')
plt.plot(x_exact, dd_exact * Z_exact, marker='o', c='black', label='Bed')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()
plt.title('Water Surface Profile')

plt.subplot(1, 3, 2)
plt.plot(x_exact, yy_pred, color="blue", label='PINN', linewidth=2)
plt.plot(x_exact, h_exact, linestyle='--', color='red', label='Exact')
plt.plot(x_exact, dd_exact * Z_exact, marker='o', c='black', label='Bed')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.ylim(ymin=1.9)
plt.ylim(ymax=2.3)
plt.xlim(xmin=7)
plt.xlim(xmax=13)
plt.legend()
plt.title('Close-Up View of Water Surface Profile')

# Plot loss convergence
plt.subplot(1, 3, 3)
plt.semilogy(history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.title('Training Loss Curve')


