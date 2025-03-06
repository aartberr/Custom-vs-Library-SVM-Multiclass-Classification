import pandas as pd
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import time

def accuracy_score(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

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
embedding_url = "https://tfhub.dev/google/nnlm-en-dim50/2"
embed = hub.load(embedding_url)

# Transform text data into embeddings
X_embedded = embed(X).numpy()

# Split data (60% training, 40% testing)
X_train, X_test, y_train, y_test = train_test_split(X_embedded, y, test_size=0.4, random_state=42)

# Experiment configurations
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
polynomial_degrees = [(1, 2), (2, 2), (1, 3), (2, 3)]  # (c, d) for polynomial kernel
C_values = [1, 1e6] #soft or hard margin
max_iterations = [1000]
tolerance_values = [1e-3]
max_accuracy = 0
best_c = None
best_d = None

# DataFrame to store all results
results_df = pd.DataFrame()

# Loop through configurations
for kernel in kernels:
    for C in C_values:
      if C == 1:
        margin = 'soft_margin'
      else:
        margin = 'hard_margin'
      for max_iter in max_iterations:
          for tol in tolerance_values:
              if kernel == 'poly':
                  for c, d in polynomial_degrees:
                      start_time = time.time()
                      # Train SVM model
                      svm_model = SVC(kernel=kernel, C=C, coef0 = c, degree=d, max_iter=max_iter, tol=tol)
                      svm_model.fit(X_train, y_train)

                      # Classify test data and evaluate
                      y_pred = svm_model.predict(X_test)

                      accuracy = accuracy_score(y_test, y_pred)

                      process_time = time.time() - start_time

                      # Append results to the DataFrame
                      results_df = pd.concat([
                           results_df,
                           pd.DataFrame([{
                              "Kernel": kernel,
                              "Margin": margin,
                              "C": C,
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

                      if accuracy > max_accuracy:
                          max_accuracy = accuracy
                          best_kernel = kernel
                          best_c = c
                          best_d = d

              else:
                  start_time = time.time()
                  # Train SVM model
                  svm_model = SVC(kernel=kernel, C=C, max_iter=max_iter, tol=tol)
                  svm_model.fit(X_train, y_train)

                  # Classify test data and evaluate
                  y_pred = svm_model.predict(X_test)

                  accuracy = accuracy_score(y_test, y_pred)

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

# Save all results to a CSV file
results_df.to_csv("svm_results_lib.csv", index=False)

print("Best Kernel:", best_kernel)

# Further Experiments with the Best Kernel
additional_results = []
max_accuracy_best_kernel = 0
for C in [0.1, 1, 10]:
    for max_iter in [1000, 5000, 10000]:
            start_time = time.time()
            # Train SVM model
            if best_kernel == 'poly':
                svm_model = SVC(kernel=best_kernel, C=C, coef0 = best_c, degree=best_d, max_iter=max_iter, tol=tol)
            else:
                svm_model = SVC(kernel=best_kernel, C=C, max_iter=max_iter, tol=tol)
            svm_model.fit(X_train, y_train)

            # Classify test data and evaluate
            y_pred = svm_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)

            process_time = time.time() - start_time

            # Append additional results
            additional_results.append({
                "C": C,
                "Polynomial_C": best_c,
                "Polynomial_D": best_d,
                "Kernel": best_kernel,
                "Max_Iter": max_iter,
                "Tolerance": tol,
                "Accuracy": accuracy,
                "Duration": process_time,
                "y_pred": list(y_pred),
                "y_test": list(y_test)
            })

            if accuracy >= max_accuracy_best_kernel:
                max_accuracy_best_kernel = accuracy
                best_C = C
                best_max_iter = max_iter

# Create a DataFrame for additional results and save it
additional_results_df = pd.DataFrame(additional_results)
additional_results_df.to_csv("svm_additional_results_lib.csv", index=False)