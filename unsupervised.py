import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

def unsupervised_kmeans(X, k_values=[2, 3, 4, 5, 6]):

    X_scaled = StandardScaler().fit_transform(X)
    
    print("\n---------------- K-Means ----------------")
    print(f"\n{'k':<5} | Silhuett-score | Optimal k?")
    print("-" * 30)
    
    best_score = -1
    best_k = None
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=1)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        
        is_best = "Yes" if score > best_score else "No"
        if score > best_score:
            best_score = score
            best_k = k
        
        print(f"{k:<5} | {score:.3f}         | {is_best}")
    
    print(f"\nOptimal number of clusters is k={best_k} (score={best_score:.3f})")



def unsupervised_dbscan(X, eps_values=[0.3, 0.5, 0.7], min_samples_values=[5, 10]):
    
    print("\n---------------- K-DBSCAN ----------------")
    print("Parameters tested:")
    print(f"eps values: {eps_values}")
    print(f"min_samples values: {min_samples_values}\n")
    
    X_scaled = StandardScaler().fit_transform(X)
    

    print(f"{'eps':<5} | {'min_samples':<12} | Number of clusters | Noise points | Silhouette score")
    print("-" * 70)
    
    for eps in eps_values:
        for min_samp in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samp)
            labels = dbscan.fit_predict(X_scaled)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            if n_clusters > 1:
                score = silhouette_score(X_scaled, labels)
            else:
                score = -1
                
            print(f"{eps:<5} | {min_samp:<12} | {n_clusters:<13} | {n_noise:<12} | {score:.3f}")
