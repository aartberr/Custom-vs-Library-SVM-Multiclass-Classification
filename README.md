# SVM Multiclass Classification – Model Comparison (Custom vs. Library-Based)  

This project was created as part of a course assignment for the **Neural Networks** class at **CSD, AUTH University**.

This project explores **multiclass text classification** using **Support Vector Machines (SVMs)** and compares a **custom-built SVM model** to a **library-based implementation**. Additionally, an **MLP with one hidden layer** is included to simulate SVM-like behavior.  

## **Implemented Methods**  

1. **Library-Based SVM Model**  
   - Utilizes an SVM implementation with **parameter tuning** for optimization.  
   - Supports multiple hyperparameters such as **margin (C), kernel type, max iterations, tolerance**, and **degree (for polynomial kernels)**.  
   - Implemented in `library_based_model.py`.  

2. **Custom SVM Model**  
   - A manually implemented SVM that follows the same structure as the library-based model.  
   - Allows direct comparison of performance and efficiency.  
   - Implemented in `scratch_model.py`.  

3. **MLP with One Hidden Layer**  
   - Included for comparison, as MLPs with a single hidden layer can approximate SVM behavior.  
   - Implemented in `mlp_1_hidden_layer.py`.  

## **Dataset Requirements**  

The models are designed for **multiclass text classification (numerical, non-ordinal labels)**. The dataset should be in CSV format and named:  

```
dataset1.csv
```

## **Key Features**  

- Evaluates **accuracy** for different parameter combinations:  
  - **Margin (C)**  
  - **Kernel type** (linear, polynomial, RBF, etc.)  
  - **Maximum iterations**  
  - **Tolerance level**  
  - **Polynomial kernel degree** (when applicable)  
- Saves accuracy and training duration results in CSV files:  
  - **`svm_results_lib.csv`** → Results from the library-based SVM model.  
  - **`svm_results_custom.csv`** → Results from the custom-built SVM model.  
- Once the best-performing model is identified, an additional search is performed over **C values** and **max iterations** to refine the results.  

## **Files**  

- **library_based_model.py** – Implements an SVM classifier using a machine learning library, allowing hyperparameter tuning.  
- **scratch_model.py** – Implements a manually built SVM classifier.  
- **mlp_1_hidden_layer.py** – Implements an MLP with one hidden layer for comparison.  
- **dataset1.csv** – The dataset file (should be provided by the user).  

## **Usage**  

1. **Run the SVM models (library-based or custom-built)**:  
   ```bash
   python library_based_model.py
   python scratch_model.py
   ```

2. **Run the MLP for comparison**:  
   ```bash
   python mlp_1_hidden_layer.py
   ```

3. **Review the results**:  
   - Open `svm_results_lib.csv` and `svm_results_custom.csv` to compare accuracy and runtime for different configurations.  

## **Future Work**  

- Improve the custom SVM implementation with further optimizations.  
- Extend to other classification tasks beyond text (e.g., image classification).  
