import matplotlib.pyplot as plt
import numpy as np

trainingDataUsage = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Perceptron Classifier Results For Face Data
pAccuracyF = [74.40, 80.93, 83.87, 85.33, 86.00, 86.00, 86.00, 86.00, 86.00, 86.00]
pstdF = [2.81, 2.30, 0.98, 0.59, 0.0, 0.01, 0.00, 0.0, 0.0, 0.0]
pTimeF = [10.296, 11.87, 13.24, 14.64, 15.76, 17.09, 18.41, 19.44, 21.20, 22.62]

# Naive Bayes Classifier Results For Face Data
nbAccuracyF = [79.60, 82.40, 86.00, 86.27, 87.20, 86.93, 89.47, 88.93, 89.60, 90.67]
nbstdF = [2.91, 2.69, 1.93, 2.09, 1.81, 1.31, 1.15, 1.61, 0.33, 0]
nbTimeF = [9.94, 9.90, 10.00, 10.10, 10.26, 10.28, 10.44, 10.45, 10.48, 10.672]


# Perceptron Classifier Results For Digit Data
pAccuracyD = [79.54, 80.62, 82.32, 82.40, 82.20, 81.70, 81.30, 81.8, 81.44, 81.0 ]
pstdD = [1.86, 0.71, 0.90, 0.78, 0.64, 0.67, 0.45, 0.59, 0.66, 0.0]
pTimeD = [45.39, 57.41, 70.22, 83.31, 97.39, 111.11, 123.73, 138.58, 152.12, 165.61 ]


# Naive Bayes Classifier Results For Digit Data
nbAccuracyD = [74.70, 75.48, 76.14, 76.40, 75.82, 76.78, 77.00,  76.04, 76.76, 77.10]
nbstdD = [0.78, 0.58, 1.00, 0.83, 0.71, 0.54, 0.37, 0.35, 0.21, 0.0]
nbTimeD = [30.37, 30.84, 30.85, 30.86, 31.52, 32.20, 32.69, 32.93, 32.92, 33.12 ]


# Visualizing the Accuracy Comparison of Naive Bayes & Perceptron Classifiers w/ Face Data
plt.figure(figsize=(10, 6))
plt.plot(trainingDataUsage, pAccuracyF, label="Perceptron", marker='o')
plt.plot(trainingDataUsage, nbAccuracyF, label="Naive Bayes", marker='o')
plt.fill_between(trainingDataUsage, np.array(pAccuracyF) - np.array(pstdF), np.array(pAccuracyF) + np.array(pstdF), alpha=0.2)
plt.fill_between(trainingDataUsage, np.array(nbAccuracyF) - np.array(nbstdF), np.array(nbAccuracyF) + np.array(nbstdF), alpha=0.2)
plt.title("Accuracy vs. Training Data Usage (Face Data)")
plt.xlabel("Training Data Usage (%)")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid()
plt.savefig("face_accuracy_comparison_plot.png")

# Visualizing the Training Time Comparison of Naive Bayes & Perceptron Classifiers w/ Face Data
plt.figure(figsize=(10, 6))
plt.plot(trainingDataUsage, pTimeF, label="Perceptron", marker='o')
plt.plot(trainingDataUsage, nbTimeF, label="Naive Bayes", marker='o')
plt.title("Training Time vs. Training Data Usage (Face Data)")
plt.xlabel("Training Data Usage (%)")
plt.ylabel("Training Time (s)")
plt.legend()
plt.grid()
plt.savefig("face_time_comparison_plot.png")

# Visualizing the Accuracy Comparison of Naive Bayes & Perceptron Classifiers w/ Digit Data
plt.figure(figsize=(10, 6))
plt.plot(trainingDataUsage, pAccuracyD, label="Perceptron", marker='o')
plt.plot(trainingDataUsage, nbAccuracyD, label="Naive Bayes", marker='o')
plt.fill_between(trainingDataUsage, np.array(pAccuracyD) - np.array(pstdD), np.array(pAccuracyD) + np.array(pstdD), alpha=0.2)
plt.fill_between(trainingDataUsage, np.array(nbAccuracyD) - np.array(nbstdD), np.array(nbAccuracyD) + np.array(nbstdD), alpha=0.2)
plt.title("Accuracy vs. Training Data Usage (Digit Data)")
plt.xlabel("Training Data Usage (%)")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid()
plt.savefig("digit_accuracy_comparison_plot.png")

# Visualizing the Training Time Comparison of Naive Bayes & Perceptron Classifiers w/ Digit Data
plt.figure(figsize=(10, 6))
plt.plot(trainingDataUsage, pTimeD, label="Perceptron", marker='o')
plt.plot(trainingDataUsage, nbTimeD, label="Naive Bayes", marker='o')
plt.title("Training Time vs. Training Data Usage (Digit Data)")
plt.xlabel("Training Data Usage (%)")
plt.ylabel("Training Time (s)")
plt.legend()
plt.grid()
plt.savefig("digit_time_comparison_plot.png")

plt.show()