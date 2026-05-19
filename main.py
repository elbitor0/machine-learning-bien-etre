import sklearn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
def load_data(xlsx_path, target_col="target"):
    bien_etre_df = pd.read_excel(xlsx_path)
    X = bien_etre_df.drop(columns=[target_col])
    Y = bien_etre_df[target_col]
    return X,Y

def normalize(X):
    standard_scaler_object = StandardScaler()
  
    X_normalized =  standard_scaler_object.fit_transform(X)
    return standard_scaler_object, X_normalized

def evaluate(X_normalized, Y, n_neighbors =5 ,n_splits=10):
    cross_validation_object = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    knn_object = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_val_score(knn_object, X_normalized, Y,cv=cross_validation_object,scoring="f1_weighted")
    
    
    print(f"f1-score moyen {scores.mean()} +/- {scores.std()}")
    return scores

def find_optimal_kvalue(X_normalized,Y,n_split,n_neighbors_range=range(1,100,2)):
    """Grid Search CV"""
    dict_of_k_scores = {}
    for n_neighbors in n_neighbors_range:
        scores = evaluate(X_normalized, Y , n_neighbors, n_splits=n_split)
        current_mean_score = scores.mean()
        dict_of_k_scores[n_neighbors] = current_mean_score
    optimal_n_neighbors = max(dict_of_k_scores, key=dict_of_k_scores.get)
    print(f"optimal n_neighbors value is {optimal_n_neighbors}")

if __name__ == "__main__":
    X, Y = load_data("bienetre.xlsx")
    standard_scaler_object, X_normalized = normalize(X)
#    print(X,Y,standard_scaler_object,X_normalized)
    print(find_optimal_kvalue(X_normalized,Y,10))

