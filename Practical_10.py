import numpy as np
import matplotlib.pyplot as plt
# In this work i decided to change the code for challenge myself
#variant №90 

# Minimaze f(X1,X2) = X1 - A * X2 + b * (X1 ** 2) + c * X1 * X2 + d * (X2 ** 2)
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
    gap_h = []
    
    #Iterative optimization algorithm
    while gap > eps and iterations < max_iter:
        grad = grad_f(X, A, b, c, d)
        X_new = X - l * grad
        gap = np.linalg.norm(X_new - X)
        X = X_new
        iterations += 1
        gap_h.append(gap)
    
    return X, gap_h

A = -2
b = 5
c = -2
d = 5
l = 0.01 #Default Lambda
eps = 1e-8
max_iter = 1000

start_point = [(9, 6), (80, 160), (200, 300)] #Starting points

optimized_points = []
gap_histories = []
for X_start in start_point:
    X_optimized, gap_h = gradient_descent(X_start, A, b, c, d, l, eps, max_iter)
    optimized_points.append(X_optimized)
    gap_histories.append(gap_h)

X1_range = np.linspace(-10, 10)#Returns num evenly spaced samples, calculated over the interval [start, stop].
X2_range = np.linspace(-10, 10)#Returns num evenly spaced samples, calculated over the interval [start stop].
X1_mesh, X2_mesh = np.meshgrid(X1_range, X2_range)
Z = f([X1_mesh, X2_mesh], A, b, c, d)

fig = plt.figure(figsize=(10, 8)) #the chart window dimensions
ax = fig.add_subplot(111, projection='3d') #IDK is it correct but i did like this and (https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html) Information from this page.
ax.plot_surface(X1_mesh, X2_mesh, Z, cmap='viridis')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('f(X1, X2)')
ax.set_title('Objective Function')

optimized_points = np.array(optimized_points)
ax.scatter(optimized_points[:, 0], optimized_points[:, 1], f(optimized_points.T, A, b, c, d), color='red', s=50, label='Optimized Points')
ax.legend()

plt.show()

plt.figure(figsize=(10, 6)) #the chart window dimensions
for i, gap_h in enumerate(gap_histories):
    plt.plot(range(len(gap_h)), gap_h, label=f'Starting Point {i+1}')
plt.xlabel('Iteration')
plt.ylabel('Optimality Gap')
plt.title('Optimality Gap vs. Iteration')
plt.legend()
plt.show()

#Draw chart lambda vs Numbers of iteration
lambdas = [0.01, 0.05 ,0.2 ,0.3, 0.4 ,0.6 ,0.8 , 1] # Array with different values to lambda
num_iterations = []

for l in lambdas:
    _, gap_h = gradient_descent(start_point[0], A, b, c, d, l, eps, max_iter)
    num_iterations.append(len(gap_h))

plt.figure(figsize=(10, 6)) #the chart window dimensions
plt.plot(lambdas, num_iterations, marker='o') # Numbers of iteration it is y and lambda it is x
plt.xlabel('Lambda')
plt.ylabel('Number of Iterations')
plt.title('Lambda vs. Number of Iterations')
plt.show()
