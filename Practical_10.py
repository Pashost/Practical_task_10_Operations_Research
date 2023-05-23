# Introduction to Operation Research
# HW X
# Jānis Pekša

import math
import matplotlib.pyplot as plt

def f(x, y):
    return x + 2 * y + 5 * (x ** 2) - 2 * x * y + 5 * (y ** 2)

def grad_f(x, y):
    return 10 * x + 1 - 2 * y, 10 * y + 2 - 2 * x

def descente(x, y, l=1e-2, eps=1e-6, maxIter=1000):
    gradx, grady = grad_f(x, y)
    grad = math.sqrt(gradx ** 2 + grady ** 2)
    i = 0
    x_vals = [x]  # List to store x values
    y_vals = [y]  # List to store y values
    while abs(grad) > eps:
        gradx, grady = grad_f(x, y)
        grad = math.sqrt(gradx ** 2 + grady ** 2)
        x = x - l * gradx
        y = y - l * grady
        i += 1
        x_vals.append(x)
        y_vals.append(y)
        if i > maxIter:
            return None
    return x, y, x_vals, y_vals

# Run gradient descent optimization
x_opt, y_opt, x_vals, y_vals = descente(5, 6)

# Plot the optimization path
plt.plot(x_vals, y_vals, 'bo-')
plt.plot(x_opt, y_opt, 'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Descent Optimization')
plt.grid(True)
plt.show()