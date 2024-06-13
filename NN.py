import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import os

def load_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    X, y = [], []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            values = line.strip().split()
            X.append(list(map(float, values[1:])))
            y.append(float(values[0]))

    return np.array(X), np.array(y)

def evaluate(X, y, feature_set, knn):
    X_selected = X[:, feature_set]
    scores = cross_val_score(knn, X_selected, y, cv=10, scoring='accuracy')
    return scores.mean()

def forward_selection(X, y, knn):
    num_features = X.shape[1]
    selected_features = []
    best_accuracy = 0.0

    for _ in range(num_features):
        best_feature = None
        best_new_accuracy = best_accuracy

        for feature in range(num_features):
            if feature not in selected_features:
                current_features = selected_features + [feature]
                accuracy = evaluate(X, y, current_features, knn)
                if accuracy > best_new_accuracy:
                    best_new_accuracy = accuracy
                    best_feature = feature

        if best_feature is not None:
            selected_features.append(best_feature)
            best_accuracy = best_new_accuracy
        else:
            break

    return selected_features, best_accuracy

def backward_elimination(X, y, knn, min_features=1, max_removed_features=100, max_selected_features=10):
    num_features = X.shape[1]
    selected_features = list(range(num_features))
    best_accuracy = evaluate(X, y, selected_features, knn)
    best_feature_set = selected_features[:]  # Track the best feature set
    
    num_removed_features = 0
    while len(selected_features) > min_features and num_removed_features < max_removed_features:
        improvement_found = False
        
        for feature in selected_features[:]:  # Iterate over a copy of selected_features
            current_features = selected_features[:]
            current_features.remove(feature)
            accuracy = evaluate(X, y, current_features, knn)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_feature_set = current_features[:]
                improvement_found = True
                num_removed_features += 1
                selected_features = current_features[:]  # Update selected_features
                break  # Exit the for loop to restart from the beginning
        
        if not improvement_found or len(selected_features) <= min_features or len(selected_features) >= max_selected_features:
            break
    
    return best_feature_set, best_accuracy


def main():
    print("Welcome to Feature Selection Algorithm")

    file_name = input("Type in the name of the file to test: ")
    try:
        X, y = load_dataset(file_name)
    except FileNotFoundError as e:
        print(e)
        return

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Hyperparameter tuning for KNN
    knn = KNeighborsClassifier()
    param_grid = {'n_neighbors': list(range(1, 31)), 'weights': ['uniform', 'distance']}
    grid_search = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
    grid_search.fit(X, y)
    best_knn = grid_search.best_estimator_

    print("Running nearest neighbor with all features, using (leaving-one-out) evaluation, I get an accuracy of:")
    accuracy = evaluate(X, y, list(range(X.shape[1])), best_knn)

    print(f"Accuracy: {accuracy:.4f}")

    print("Starting feature search:")

    # Forward Selection
    print("(1) Forward Selection")
    selected_features, best_accuracy = forward_selection(X, y, best_knn)
    print(f"Best feature set: {selected_features}")
    print(f"Best accuracy: {best_accuracy:.4f}")

    # Backward Elimination
    print("(2) Backward Elimination")
    selected_features, best_accuracy = backward_elimination(X, y, best_knn, min_features=1, max_removed_features=100, max_selected_features=10)
    print(f"Best feature set: {selected_features}")
    print(f"Best accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    main()
