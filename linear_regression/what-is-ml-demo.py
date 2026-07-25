import numpy as np
import matplotlib.pyplot as plt

# Two training points
x = np.array([1, 2])
y = np.array([2, 3])   # encodes y = x + 1

# Fit linear regression: y = m*x + b
m, b = np.polyfit(x, y, 1)

# Predict at x = 42
x_future = 42
y_future = m * x_future + b

print("Slope (m):", m)
print("Intercept (b):", b)
print(f"Prediction at x={x_future}: y={y_future}")

# Plot training data
plt.scatter(x, y, color='blue', label='Training Data')

# Plot trendline
x_line = np.linspace(min(x), x_future, 200)
y_line = m * x_line + b
plt.plot(x_line, y_line, color='red', label=f"Trendline: y = {m:.2f}x + {b:.2f}")

# Plot prediction point
plt.scatter([x_future], [y_future], color='green', s=80, label=f"Predicted (42, {y_future:.2f})")

plt.title("Linear Regression with Two Points")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
