import numpy as np
import matplotlib.pyplot as plt
# In this work i decided to change the code for chalenge myself

def f(X, A, b, c, d):
    X1, X2 = X
    return X1 - A * X2 + b * (X1 ** 2) + c * X1 * X2 + d * (X2 ** 2)

def grad_f(X, A, b, c, d):
    X1, X2 = X
    grad_X1 = 1 + 2 * b * X1 + c * X2
    grad_X2 = -A + c * X1 + 2 * d * X2
    return np.array([grad_X1, grad_X2])

def gradient_descent(X_init, A, b, c, d, l=0.01, eps=1e-6, max_iter=1000):
    X = np.array(X_init) 
    gap = np.inf
    iterations = 0
    gap_history = []
    
    while gap > eps and iterations < max_iter:
        grad = grad_f(X, A, b, c, d)
        X_new = X - l * grad
        gap = np.linalg.norm(X_new - X)
        X = X_new
        iterations += 1
        gap_history.append(gap)
    
    return X, gap_history

A = -2
b = 5
c = -2
d = 5
l = 0.01
eps = 1e-6
max_iter = 1000

X_starts = [(10, 5), (-7, 1), (8, -9)] 

optimized_points = []
gap_histories = []

for X_start in X_starts:
    X_optimized, gap_history = gradient_descent(X_start, A, b, c, d, l, eps, max_iter)
    optimized_points.append(X_optimized)
    gap_histories.append(gap_history)

X1_range = np.linspace(-10, 10, 100)
X2_range = np.linspace(-10, 10, 100)
X1_mesh, X2_mesh = np.meshgrid(X1_range, X2_range)
Z = f([X1_mesh, X2_mesh], A, b, c, d)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1_mesh, X2_mesh, Z, cmap='viridis')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('f(X1, X2)')
ax.set_title('Objective Function')

optimized_points = np.array(optimized_points)
ax.scatter(optimized_points[:, 0], optimized_points[:, 1], f(optimized_points.T, A, b, c, d), color='red', s=50, label='Optimized Points')
ax.legend()

plt.show()

plt.figure(figsize=(10, 6))
for i, gap_history in enumerate(gap_histories):
    plt.plot(range(len(gap_history)), gap_history, label=f'Starting Point {i+1}')
plt.xlabel('Iteration')
plt.ylabel('Optimality Gap')
plt.title('Optimality Gap vs. Iteration')
plt.legend()
plt.show()

lambdas = [0.001, 0.01, 0.1, 1]
num_iterations = []

for l in lambdas:
    _, gap_history = gradient_descent(X_starts[0], A, b, c, d, l, eps, max_iter)
    num_iterations.append(len(gap_history))

plt.figure(figsize=(10, 6))
plt.plot(lambdas, num_iterations, marker='o')
plt.xlabel('Lambda')
plt.ylabel('Number of Iterations')
plt.title('Lambda vs. Number of Iterations')
plt.show()
