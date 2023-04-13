import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd

class KMeans:
    def __init__(self, k, max_iter=100, tol=1e-4):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None

    def initialize_centroids(self, data):
        indices = np.random.permutation(data.shape[0])
        self.centroids = data[indices[:self.k]]

    def calculate_closest_clusters(self, data, centroids):
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def compute_centroids(self, data, labels):
        new_centroids = np.zeros((self.k, data.shape[1]))
        for i in range(self.k):
            new_centroids[i] = np.mean(data[labels == i], axis=0)
        return new_centroids

    def objective(self, data):
        total_distance = 0
        for i in range(self.centroids.shape[0]):
            total_distance += np.sum(np.linalg.norm(data[self.labels == i] - self.centroids[i], axis=1))
        return total_distance

    def plot_clusters(self, data, centroids, iteration):
        plt.scatter(data[:, 0], data[:, 1], c=self.labels, cmap='viridis', marker='o', s=100, edgecolor='k')
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, edgecolor='k')
        plt.title(f'Clusters at iteration {iteration}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

    def fit(self, data):
        objective_values = []
        self.initialize_centroids(data)
        for i in range(self.max_iter):
            print(f"Iteration number: {i+1}")
            old_centroids = self.centroids
            self.labels = self.calculate_closest_clusters(data, old_centroids)
            self.centroids = self.compute_centroids(data, self.labels)
            objective_values.append(self.objective(data))

            if (i == 0 or i == 2) :
                self.plot_clusters(data, old_centroids, i+1)

            if np.linalg.norm(self.centroids - old_centroids) < self.tol:
                break
        
       
        self.plot_clusters(data, self.centroids, 'converged')

        x_axis = np.linspace(1, len(objective_values), len(objective_values))
        if self.k == 3:
            plt.plot(x_axis, objective_values)
            plt.xlabel('Iteration Number')
            plt.xticks(x_axis)
            plt.ylabel('Objective Function')
            plt.title('Error Across Iterations', fontweight="bold")
            plt.show()

        return self.labels
    
    def plot_decision_boundaries(self, data, resolution=100):
        x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
        y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                             np.linspace(y_min, y_max, resolution))

        Z = self.calculate_closest_clusters(np.c_[xx.ravel(), yy.ravel()], self.centroids)
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
        plt.scatter(data[:, 0], data[:, 1], c=self.labels, cmap='viridis', marker='o', s=100, edgecolor='k')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='x', s=200, edgecolor='k')
        plt.title('Decision Boundaries and Clusters')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
    


iris = pd.read_csv('irisdata.csv')
iris = iris.iloc[:,[2,3]].values  

# Apply k-means clustering for k = 2 and k = 3
for k in [2, 3]:
    print(f"K-means clustering with k = {k}")
    kmeans = KMeans(k)
    labels = kmeans.fit(iris)
    kmeans.plot_decision_boundaries(iris)
