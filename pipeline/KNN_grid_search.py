
import os
import numpy as np
import pandas as pd 

from zadu.measures import *

from pipeline.utils.embedding_utils import (loading_embeddings,
                                            save_embeddings_to_zarr,
                                            preprocessing_embeddings,
                                            eval_cluster,
                                            caclualte_medoid_centroid_for_clusters
                                            )

from datetime import datetime


import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pickle 


X_train_filename = 'data/database/knn_data/X_train.npy'
X_test_filename = 'data/database/knn_data/X_test.npy'
y_train_filename = 'data/database/knn_data/y_train.npy'
y_test_filename = 'data/database/knn_data/y_test.npy'

if os.path.exists(X_train_filename) and os.path.exists(X_test_filename):
    print("Loading existing train and test data...")
    X_train = np.load(X_train_filename)
    X_test = np.load(X_test_filename)
    y_train = np.load(y_train_filename)
    y_test = np.load(y_test_filename)

else:
# loading the Y and X.
    df = pd.read_csv("data/database/embeddinngs_mpnet_task_dedup.csv")
    embeddings = loading_embeddings("data/database/embeddinngs_mpnet_task_dedup.zarr")

    X = embeddings 
    Y = df["Task| cluster_id"].to_numpy()

    assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    
    # GridSearch hyperparameters for 

    # 2. Use numpy.save() to save the arrays to files
    np.save(X_train_filename, X_train)
    np.save(X_test_filename, X_test)
    np.save(y_train_filename, y_train)
    np.save(y_test_filename, y_test)
    
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


# Define the parameter grid
param_grid = {
    'n_neighbors': [5, 10, 20, 30, 50],
    'metric': ['euclidean', 'manhattan', 'cosine', 'minkowski'],
    'weights': ['uniform', 'distance']
}

# Create the k-NN classifier
knn = KNeighborsClassifier()


# Perform GridSearchCV
grid_search = GridSearchCV(knn, param_grid, cv=10, scoring='f1_macro', verbose=3)  # cv=5 means 5-fold cross-validation

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and the best score
print("Best hyperparameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Get the best estimator (the k-NN classifier with the best hyperparameters)
best_knn = grid_search.best_estimator_

# Evaluate the best estimator on the test set
f1_macro = best_knn.score(X_test, y_test)
print("f1 macro:", f1_macro)


# 2. Precision, Recall, and F1-score: These are more informative, especially with imbalanced classes.
#    * Precision:  Out of all the instances predicted as positive, how many were actually positive?
#    * Recall: Out of all the actual positive instances, how many were correctly predicted as positive?
#    * F1-score: The harmonic mean of precision and recall.

# precision = precision_score(y_test, y_pred, average='weighted') # Use 'weighted' for multi-class
# recall = recall_score(y_test, y_pred, average='weighted')       # Use 'weighted' for multi-class
# f1 = f1_score(y_test, y_pred, average='weighted')               # Use 'weighted' for multi-class
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1-score: {f1:.4f}")

# report = classification_report(y_test, y_pred)
# print("Classification Report:")
# print(report)


# Best hyperparameters: {'metric': 'cosine', 'n_neighbors': 10, 'weights': 'distance'}
# Best score: 0.9444179624309861
# f1 macro: 0.9528373266078184

time_str = datetime.now().strftime("%d%m%Y")

with open(f"output/knn_classifiers/best_knn_{time_str}.pkl", "wb") as f:
    pickle.dump(best_knn, f)

