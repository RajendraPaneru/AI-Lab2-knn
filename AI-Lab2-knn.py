import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import make_classification

# -------------------------------
# Step 1: Create simple 2D dataset
# -------------------------------
def generate_data():
    return np.array([
        [1, 2], [2, 3], [3, 3],     # Class 0
        [6, 5], [7, 7], [8, 6]      # Class 1
    ]), np.array([0, 0, 0, 1, 1, 1])

# -------------------------------
# Step 2: Define Distance Function
# -------------------------------
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# -------------------------------
# Step 3: KNN Algorithm
# -------------------------------
def knn(data, labels, query_point, k):
    distances = [euclidean_distance(point, query_point) for point in data]
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = labels[k_indices]
    vote = Counter(k_nearest_labels).most_common(1)
    return vote[0][0], k_indices

# -------------------------------
# Step 4: Ask user for input
# -------------------------------
try:
    x = float(input("Enter X coordinate: "))
    y = float(input("Enter Y coordinate: "))
    k = int(input("Enter value of K: "))
    user_point = np.array([x, y])
except ValueError:
    print("Invalid input. Please enter numeric values.")
    exit()

# -------------------------------
# Step 5: Run the KNN
# -------------------------------
data, labels = generate_data()
predicted_class, neighbors = knn(data, labels, user_point, k)

# -------------------------------
# Step 6: Plot the result
# -------------------------------
colors = ['blue', 'red']
for label in np.unique(labels):
    plt.scatter(data[labels == label][:, 0], data[labels == label][:, 1],
                label=f"Class {label}", color=colors[label])

plt.scatter(user_point[0], user_point[1], color='green', s=200, edgecolor='black', label='Input Point')
plt.scatter(data[neighbors][:, 0], data[neighbors][:, 1], facecolors='none', edgecolors='black', linewidths=2, s=150)

plt.title(f"KNN Classification Result: Class {predicted_class}")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()
