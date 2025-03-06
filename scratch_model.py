# SVM from scratch
import numpy as np
import time
import pandas as pd
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from sklearn.model_selection import train_test_split

def find_alpha(X, y, soft_or_hard_margin, Kernel_matrix, C, learning_rate = 0.01, max_iter = 1000, tol = 1e-6):
  n_samples = len(y)
  alpha = np.zeros(n_samples)

  # Solve the optimization problem
  for _ in range(max_iter):
    # Compute the gradient
    gradient = np.ones(n_samples) - (alpha * y) @ (Kernel_matrix * np.outer(y, y))

    # Gradient ascent step
    alpha += learning_rate * gradient

    if soft_or_hard_margin == 'hard_margin':
      # Project alpha onto the feasible region (only non-negative values)
      alpha = np.maximum(alpha, 0)  # Ensure alpha >= 0
    elif soft_or_hard_margin == 'soft_margin':
      # Project alpha onto the feasible region
      alpha = np.clip(alpha, 0, C)

    # Ensure the equality constraint: sum(alpha * y) = 0
    alpha -= y * (y @ alpha) / np.sum(y**2)

    # Convergence check
    if np.linalg.norm(gradient, ord=1) < tol:
      break

  return alpha

def decision_function(alpha, x_new, X, y, C, c, d, kernel='linear',  tol=1e-6):
    x_new = np.atleast_2d(x_new)  # x_new is treated as 2D

    # Identify support vectors
    support_indices = np.where(alpha > tol)[0]  # Tolerance for non-zero alpha
    #support_vectors -> X[support_indices]
    support_labels = y[support_indices]

    # Compute b (bias) using support vectors
    if kernel == 'linear':
        # Compute w for linear kernel
        w = np.sum((alpha * y)[:, np.newaxis] * X, axis=0)

        # Compute bias b for linear kernel
        b_values = []
        for i in support_indices:
            b_i = y[i] - np.dot(w, X[i])
            b_values.append(b_i)
        b = np.mean(b_values)

        # Decision function for linear kernel: dot(x_new, w) + b
        return np.sign(np.dot(x_new, w) + b)

    else:
        # Decision function for non-linear kernels 
        predictions = []
        for x_test in x_new:
            result = 0
            # Sum over the support vectors and apply the kernel function
            for i in support_indices: 
                if kernel == 'polynomial':
                    kernel_value = (np.dot(x_test, X[i]) + c) ** d
                elif kernel == 'rbf':
                    gamma = 1.0 / X.shape[1]  # Default gamma
                    squared_dist = np.sum(x_test**2) + np.sum(X[i]**2) - 2 * np.dot(x_test, X[i])
                    kernel_value = np.exp(-gamma * squared_dist)
                elif kernel == 'sigmoid':
                    gamma = 1.0 / X.shape[1]  # Default gamma
                    kernel_value = np.tanh(gamma * np.dot(x_test, X[i]))

                result += alpha[i] * y[i] * kernel_value

            # Compute bias b for non-linear kernel
            b_values = []
            for i in support_indices:
                b_i = y[i] - np.sign(result)
                b_values.append(b_i)
            b = np.mean(b_values)

            predictions.append(np.sign(result + b))

        return np.array(predictions)


# Make an svm class
class SVM:
  def __init__(self, kernel='linear'):
    self.alphas = []
    self.classes = [] # Each alpha is stored with the class it predicts
    self.kernel = kernel
    self.Kernel_matrix = None
    self.tol = None
    self.C = None
    self.c = None
    self.d = None

  def train(self, X, y, soft_or_hard_margin, C, c, d, learning_rate = 0.01, max_iter = 1000, tol = 1e-6): # Compute the hyperplane that seperates the data
    # Transform y to np array
    y = y.values
    # Get all the class labels
    class_labels = np.unique(y)
    # Save values
    self.tol = tol
    self.C = C
    self.c = c
    self.d = d

    # Apply the kernel
    if self.kernel == 'linear':
      self.Kernel_matrix = np.dot(X, X.T)
    elif self.kernel == 'polynomial':
      self.Kernel_matrix = (np.dot(X, X.T) + c)**d
    elif self.kernel == 'rbf':
      gamma = 1.0 / X.shape[1] # Same default as sklearn SVC for comparison reasons  
      squared_distances = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
      self.Kernel_matrix = np.exp(-gamma * squared_distances)
    elif self.kernel == 'sigmoid':
      gamma =  1.0 / X.shape[1]
      self.Kernel_matrix = np.tanh(gamma * np.dot(X, X.T)) # Same default as sklearn SVC for comparison reasons  


    # Apply 1 versus all for each class (to find its parameter alpha)
    for class_label in class_labels:
      print(f">>>>class: {class_label}, kernel = {self.kernel}")
      one_vs_all_y = np.array([-1 if label == class_label else 1 for label in y])

      # Solve the optimization problem
      alpha = find_alpha(X, one_vs_all_y, soft_or_hard_margin, self.Kernel_matrix, C, learning_rate, max_iter, tol)
      # append to alpha and save the class name
      self.alphas.append(alpha)
      self.classes.append(class_label)

  def predict(self, X_test, X_train, y_train): # Make a predict function that classifies the input data
    y_train = y_train.values
    predictions = []

    # Recursive helper function to predict the class for a single test sample
    def predict_recursive(x_test, index=0):
      if index >= len(self.classes):  # If we've checked all classes return the last class
        return self.classes[-1]

      class_label = self.classes[index]
      alpha = self.alphas[index]

      # Apply 1 versus all for each class
      one_vs_all_y = np.array([-1 if label == class_label else 1 for label in y_train])

      # Use the decision function to get the prediction for the current class
      decision = decision_function(alpha, x_test, X_train, one_vs_all_y, self.C, self.c, self.d ,self.kernel, self.tol)

      if decision == -1:
        return class_label

      # Otherwise, recursively call the function with the next class
      return predict_recursive(x_test, index + 1)

    # Loop through each test sample and apply the recursive prediction
    for x_test in X_test:
      found_class = predict_recursive(x_test)
      predictions.append(found_class)

    return predictions

  def evaluate(self, X_test, y_test, X_train, y_train): # Make a accuracy function that uses predict
    predictions = self.predict(X_test, X_train, y_train)
    accuracy = np.sum(y_test == predictions) / len(y_test)
    return accuracy, predictions

# Load dataset
df = pd.read_csv('dataset1.csv')

# Preprocess data
# Ensure no missing values
df = df.dropna()
df = df.head(100000)

# Find the number of samples (N) in the smallest class
class_counts = df['label'].value_counts()
N = class_counts.min()

# From each class keep only N samples
df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(N, random_state=42))

class_counts = df['label'].value_counts()

print(f"class counts: {class_counts}")

# Separate features and labels
X = df['text']  # Text data
y = df['label']  # Labels

# Load the pre-trained embedding model from TensorFlow Hub
embedding_url = "https://www.kaggle.com/models/google/nnlm/tensorFlow2/en-dim50/1?tfhub-redirect=true"
embed = hub.load(embedding_url)

# Transform text data into embeddings
X_embedded = embed(X).numpy()


# Split data (60% training, 40% testing)
X_train, X_test, y_train, y_test = train_test_split(X_embedded, y, test_size=0.4, random_state=42)

# Experiment configurations
kernels = ['linear', 'polynomial', 'rbf', 'sigmoid']
polynomial_degrees = [(1, 2), (2, 2), (1, 3), (2, 3)]  # (c, d) for polynomial kernel
C_values = [1] 
max_iterations = [1000]
tolerance_values = [1e-3]
max_accuracy = 0

# DataFrame to store all results
results_df = pd.DataFrame()

# Loop through configurations
for kernel in kernels:
  for C in C_values:
    for margin in ['soft_margin', 'hard_margin']:
      for max_iter in max_iterations:
          for tol in tolerance_values:
              if kernel == 'polynomial': 
                  for c, d in polynomial_degrees:
                      start_time = time.time() 
                      # Train SVM model
                      svm_model = SVM(kernel=kernel)
                      svm_model.train(X_train, y_train, margin, C=C, c=c, d=d, learning_rate = 0.01, max_iter = max_iter, tol = tol)
                      # Classify test data and evaluate
                      accuracy, y_pred = svm_model.evaluate(X_test, y_test, X_train, y_train)

                      process_time = time.time() - start_time

                      # Append results to the DataFrame
                      results_df = pd.concat([
                           results_df,
                           pd.DataFrame([{
                              "Kernel": kernel,
                              "Margin": margin,
                              "C": None,
                              "Polynomial_C": c,
                              "Polynomial_D": d,
                              "Max_Iter": max_iter,
                              "Tolerance": tol,
                              "Duration": process_time,
                              "Accuracy": accuracy,
                              "y pred": list(y_pred),
                              "y_test": list(y_test)
                           }])
                      ], ignore_index=True)

              else:
                  start_time = time.time() 
                  # Train SVM model
                  svm_model = SVM(kernel=kernel)
                  svm_model.train(X_train, y_train, margin, C=C, c=None, d=None, learning_rate = 0.01, max_iter = max_iter, tol = tol)
                  # Classify test data and evaluate
                  accuracy, y_pred = svm_model.evaluate(X_test, y_test, X_train, y_train)

                  process_time = time.time() - start_time

                  # Append results to the DataFrame
                  results_df = pd.concat([
                     results_df,
                     pd.DataFrame([{
                          "Kernel": kernel,
                          "Margin": margin,
                          "C": C,
                          "Polynomial_C": None,
                          "Polynomial_D": None,
                          "Max_Iter": max_iter,
                          "Tolerance": tol,
                          "Duration": process_time,
                          "Accuracy": accuracy,
                          "y pred": list(y_pred),
                          "y_test": list(y_test)
                     }])
                 ], ignore_index=True)

              if accuracy > max_accuracy:
                  max_accuracy = accuracy
                  best_kernel = kernel
                  best_margin = margin
                  if best_kernel == 'polynomial':
                      best_d = d
                      best_c = c
                  else:
                      best_d = None
                      best_c = None

# Save all results to a CSV file
results_df.to_csv("svm_results_scratch.csv", index=False)

# Analyze results to find the best kernel
best_kernel = results_df.loc[results_df['Accuracy'].idxmax(), 'Kernel']
print("Best Kernel:", best_kernel)

# Further Experiments with the Best Kernel
additional_results = []
max_accuracy_best_kernel = 0
for C in [0.1, 1, 10]:
    for max_iter in [1000, 2000]:
            start_time = time.time()
            # Train SVM model
            svm_model = SVM(kernel=best_kernel)
            svm_model.train(X_train, y_train, best_margin, C=C, c=best_c, d=best_d, learning_rate = 0.01, max_iter = max_iter, tol = tol)
            # Classify test data and evaluate
            accuracy, y_pred = svm_model.evaluate(X_test, y_test, X_train, y_train)

            process_time = time.time() - start_time
            
            # Append additional results
            additional_results.append({
                "C": C,
                "d": best_d,
                "Kernel": best_kernel,
                "Max_Iter": max_iter,
                "Tolerance": tol,
                "Accuracy": accuracy,
                "Duration": process_time,
                "y_pred": list(y_pred),
                "y_test": list(y_test)
            })

            if accuracy > max_accuracy_best_kernel:
                max_accuracy_best_kernel = accuracy
                best_C = C
                best_max_iter = max_iter

# Create a DataFrame for additional results and save it
additional_results_df = pd.DataFrame(additional_results)
additional_results_df.to_csv("svm_additional_results_scratch.csv", index=False)