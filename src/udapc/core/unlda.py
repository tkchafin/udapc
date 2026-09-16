import os 
import sys 

import random
import numpy as np
import seaborn as sns
import pandas as pd
import scipy

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from scipy.linalg import eigh
from scipy.linalg import sqrtm
from scipy.linalg import fractional_matrix_power

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.linalg import eigh
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


class UnRTLDA:
    def __init__(self, c, Ninit=10, gamma=1e-6, tol=1e-6, max_iter=100, Ntry=10, center=True, no_pca=False):
        self.c = c
        self.Ninit = Ninit
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.Ntry = Ntry
        self.center = center
        self.no_pca = no_pca
        self.W = None
        self.St = None
        self.labels_ = None
        self.obj = None
        self.coords_ = None
        self.cluster_centers_ = None

    def fit(self, X):
        """
        Fit the Un-RTLDA model to the input data.
        
        Args:
            X (numpy array): Input data of shape (n_samples, n_features).
        Returns:
            self: Fitted UnRTLDA instance.
        """
        n, d = X.shape

        # Center the data if needed
        if self.center:
            H = np.eye(n) - np.ones((n, n)) / n
        else:
            H = np.eye(n)

        # Compute the within-class scatter matrix St
        self.St = X.T @ H @ X
        Stt = self.St + self.gamma * np.eye(d)  # Add regularization term to St

        it = 0

        obj_old = -np.inf
        obj_new = 0.0
        Ypre = None
        T_old = None
        W2_old = None
        cluster_centers = None
        T = None

        # Initialize W using PCA 
        m = min(d, self.c - 1)
        if self.no_pca:
            W = X.T[:, :m]
        else:
            pca = PCA(n_components=m)
            W = pca.fit_transform(X.T @ H)
        W2 = W

        obj_log = []

        # Iterate until convergence or maxIter is reached
        while (not np.isclose(obj_old, obj_new, atol=self.tol) or it == 0) and it < self.max_iter:
            it += 1
            obj_old = obj_new

            # Calculate the intermediate matrix product
            T = (scipy.linalg.expm(-0.5 * np.linalg.inv(W2.T @ Stt @ W2)) @ W2.T @ X.T @ H).T

            best_obj_tmp = float('inf')
            best_Ypre = None

            # Loop through Ntry times to find the best clustering
            for j in range(self.Ntry):
                kmeans = KMeans(n_clusters=self.c, tol=self.tol, max_iter=self.max_iter, n_init=self.Ninit)  # Initialize KMeans clustering
                Ypre_temp = kmeans.fit_predict(T)  # Cluster the data and obtain labels
                obj_tmp = kmeans.inertia_  # Store the within-cluster sum of squares
                # Update Ypre if the new clustering is better than the previous one
                if obj_tmp < best_obj_tmp:
                    best_obj_tmp = obj_tmp
                    best_Ypre = Ypre_temp
                    cluster_centers = kmeans.cluster_centers_
            Ypre = best_Ypre

            # Update Yp matrix
            Yp = np.eye(self.c)[Ypre]

            # Compute the between-class scatter matrix Sb
            Sb = X.T @ H @ Yp @ np.linalg.inv(Yp.T @ Yp) @ Yp.T @ H.T @ X

            # Perform generalized eigenvalue decomposition and update W2
            # eigvals, eigvecs = eigh(Sb, Stt)
            # W2 = eigvecs[:, -m:]
            model = gevd(Sb, Stt)
            W2 = model['W'][:, -m:]

            # Update the new objective value
            obj_new = np.trace((W2.T @ Stt @ W2) ** -1 @ W2.T @ Sb @ W2)
            obj_log.append(obj_new)

        if it == self.max_iter:
            print(f"Warning: The un_rtlda did not converge within {self.max_iter} iterations!")

        self.labels_ = Ypre
        self.obj = obj_log 
        self.coords_ = T
        self.W = W2
        self.cluster_centers_ = cluster_centers

        return self.coords_, self.labels_

    def transform(self, X):
        """
        Transform the input data using the learned W matrix.
        
        Args:
            X (numpy array): Input data of shape (n_samples, n_features).
        Returns:
            T (numpy array): Transformed data of shape (n_samples, n_components).
        """
        # Get the number of samples and features from the input data
        n, d = X.shape
        
        # Check if centering is required and compute the centering matrix H accordingly
        if self.center:
            H = np.eye(n) - np.ones((n, n)) / n
        else:
            H = np.eye(n)
        
        # Compute the regularized within-class scatter matrix Stt
        Stt = self.St + self.gamma * np.eye(d)
        
        # Apply the learned W matrix to transform the input data
        T = (scipy.linalg.expm(-0.5 * np.linalg.inv(self.W.T @ Stt @ self.W)) @ self.W.T @ X.T @ H).T

        return T
    
    def predict(self, X):
        """
        Predict the cluster labels for the input data.

        Args:
            X (numpy array): Input data of shape (n_samples, n_features).
        Returns:
            G (numpy array): Cluster labels for the input data.
        """
        # Transform the input data using the learned W matrix
        T = self.transform(X)

        # Calculate the Euclidean distances between the transformed samples and the cluster centroids
        distances = scipy.spatial.distance.cdist(T, self.cluster_centers_)

        # Assign each sample to the cluster with the nearest centroid
        G = np.argmin(distances, axis=1)

        return G

def generate_synthetic_data(n_samples=1000, n_clusters=4, n_features=50, random_state=None, dispersion=1):
    """
    Generate synthetic data with specified number of samples, clusters, and features.
    
    Args:
        n_samples (int, optional): The number of samples in the generated dataset. Defaults to 1000.
        n_clusters (int, optional): The number of clusters in the generated dataset. Defaults to 4.
        n_features (int, optional): The number of features in the generated dataset. Defaults to 50.
        random_state (int, optional): The random seed for reproducibility. Defaults to None.
        dispersion (float, optional): The dispersion of the clusters. Controls the standard deviation of the clusters.
                                      Defaults to 1.
                                      
    Returns:
        tuple: A tuple containing the generated data (numpy array) and the corresponding labels (numpy array).
    """
    # Define the minimum and maximum standard deviation for the clusters
    min_std = 0.1
    max_std = 1
    
    # Calculate the standard deviation for the clusters based on the dispersion parameter
    cluster_std = min_std + (max_std - min_std) * dispersion
    
    # Generate synthetic data with `n_clusters` clusters
    data, labels = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=n_features, 
                              random_state=random_state, cluster_std=cluster_std)
    
    # Standardize the features
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    return data, labels



def gevd(A, B):
    """
    Generalized eigendecomposition of two symmetric square matrices A and B.
    
    Args:
        A (numpy array): A symmetric square matrix of shape (n, n).
        B (numpy array): A symmetric square matrix of shape (n, n).
        
    Returns:
        dict: A dictionary containing the sorted eigenvectors ('W') and a diagonal matrix of the sorted eigenvalues ('D').
    """
    # Compute the generalized eigenvectors and eigenvalues
    eigvals, eigvecs = eigh(A, B)
    
    # Sort the eigenvalues and eigenvectors in ascending order
    ind = np.argsort(eigvals)
    eigvals_sorted = eigvals[ind]
    eigvecs_sorted = eigvecs[:, ind]
    
    # Create a dictionary with the sorted eigenvectors and eigenvalues
    model = {'W': eigvecs_sorted, 'D': np.diag(eigvals_sorted)}
    
    return model

def is_positive_definite(matrix):
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

# Define the Un-RTLDA function
def un_rtlda(X, c, Ninit=10, gamma=1e-6, tol=1e-6, max_iter=100, Ntry=10, center=True, no_pca=False):
    """
    Implement the Un-Regularized Two-Level Discriminant Analysis (Un-RTLDA) algorithm for clustering.

    Args:
        X (numpy array): Input data of shape (n_samples, n_features).
        c (int): Number of clusters.
        Ninit (int, optional): Number of initializations for KMeans. Defaults to 10.
        gamma (float, optional): Regularization parameter for the within-class scatter matrix. Defaults to 1e-6.
        tol (float, optional): Convergence tolerance. Defaults to 1e-6.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        Ntry (int, optional): Number of attempts to find the best clustering. Defaults to 10.
        center (bool, optional): Whether to center the data. Defaults to True.
        no_pca (bool, optional): Whether to disable PCA initialization. Defaults to False.

    Returns:
        T (numpy array): Un-RTLDA embeddings of shape (n_samples, n_components).
        Ypre (list): Cluster assignments for each sample.
        W2 (numpy array): Eigenvectors matrix of shape (n_features, n_components).
    """
    n, d = X.shape  # Number of samples
    
    if center:
        H = np.eye(n) - np.ones((n, n)) / n 
    else:
        H = np.eye(n)

    St = X.T @ H @ X  # Compute the within-class scatter matrix St
    Stt = St + gamma * np.eye(d)  # Add regularization term to St

    it = 0  # Initialize the iteration counter
    
    obj_old = -np.inf  # Initialize the old objective value
    obj_new = 0.0 # Initialize the new objective value
    Ypre = None  # Initialize the predicted cluster labels
    T_old = None
    W2_old = None
    T = None

    # Initialize W using PCA 
    m = min(d, c - 1)
    pca = PCA(n_components=m)

    if no_pca:
        W = X.T[:, :m]
    else:
        W = pca.fit_transform(X.T @ H)
    W2 = W  # Initialize W2 with W

    obj_log = []

    # Iterate until convergence or maxIter is reached
    while (not np.isclose(obj_old, obj_new, atol=tol) or it == 0) and it < max_iter:

        it += 1
        obj_old = obj_new

        # Calculate the intermediate matrix product
        T = (scipy.linalg.expm(-0.5 * np.linalg.inv(W2.T @ Stt @ W2)) @ W2.T @ X.T @ H).T
        #T = (fractional_matrix_power(W2.T @ Stt @ W2, -0.5) @ W2.T @ X.T @ H).T

        best_obj_tmp = float('inf')
        best_Ypre = None

        # Loop through Ntry times to find the best clustering
        for j in range(Ntry):
            kmeans = KMeans(n_clusters=c, tol=tol, max_iter=max_iter, n_init=Ninit)  # Initialize KMeans clustering
            Ypre_temp = kmeans.fit_predict(T)  # Cluster the data and obtain labels
            obj_tmp = kmeans.inertia_  # Store the within-cluster sum of squares
            # Update Ypre if the new clustering is better than the previous one
            if obj_tmp < best_obj_tmp:
                best_obj_tmp = obj_tmp
                best_Ypre = Ypre_temp
        Ypre = best_Ypre
    
        # Update Yp matrix
        Yp = np.eye(c)[Ypre]

        # Compute the between-class scatter matrix Sb
        Sb = X.T @ H @ Yp @ np.linalg.inv(Yp.T @ Yp) @ Yp.T @ H.T @ X

        # Perform generalized eigenvalue decomposition and update W2
        model = gevd(Sb, Stt)
        W2 = model['W'][:, -m:]

        # Update the new objective value
        obj_new = np.trace((W2.T @ Stt @ W2) ** -1 @ W2.T @ Sb @ W2)

        obj_log.append(obj_new)

    # Print a warning if the algorithm did not converge within maxIter iterations
    if it == max_iter:
        print(f"Warning: The un_rtlda did not converge within {max_iter} iterations!")

    return T, Ypre, W2, obj_log

def plot_embeddings(T, G, W, dataset, labels, n_clusters, filename="embeddings_plots.pdf", no_pca=False):
    """
    Generate scatter plots of PCA embeddings and Un-RTLDA embeddings with cluster assignments and original population labels.
    
    Args:
        T (numpy array): Un-RTLDA embeddings of shape (n_samples, n_components).
        G (list): Cluster assignments for each sample.
        W (numpy array): Eigenvectors matrix of shape (n_features, n_components).
        dataset (numpy array): Original dataset of shape (n_samples, n_features).
        labels (list): Original population labels for each sample.
        n_clusters (int): Number of clusters in the data.
        filename (str, optional): Name of the output PDF file containing the plots. Defaults to "embeddings_plots.pdf".
    """
    if no_pca:
        X = dataset[:, :2]
    else:
        pca = PCA(n_components=2)
        X = pca.fit_transform(dataset)
    
    # Create a DataFrame for plotting
    df = pd.DataFrame(X, columns=[f"PC{i+1}" for i in range(X.shape[1])])
    df["Cluster"] = G
    df["Original_Population"] = labels

   
    df2 = pd.DataFrame(T, columns=[f"DA{i+1}" for i in range(T.shape[1])])
    df2["Cluster"] = G
    df2["Original_Population"] = labels


    with PdfPages(filename) as pdf:
            # Create a scatter plot for PCA embeddings with Un-RTLDA cluster assignments and original population labels
            plt.figure()
            sns.scatterplot(data=df, x="PC1", y="PC2", hue="Cluster", style="Original_Population", palette="deep")
            plt.title("Un-RTLDA.v2 Clusters on PCA Embeddings")
            pdf.savefig()
            plt.close()

            # Create a scatter plot for Un-RTLDA embeddings with cluster assignments and original population labels
            
            if T.shape[1] > 1:
                plt.figure()
                # Create a scatter plot for Un-RTLDA embeddings with cluster assignments and original population labels
                sns.scatterplot(data=df2, x="DA1", y="DA2", hue="Cluster", style="Original_Population", palette="deep")
                plt.title("Un-RTLDA.v2 Embeddings")
            else:
                # Create a KDE plot for the case when there's only one DA axis
                plt.figure()
                sns.kdeplot(x="DA1", hue="Cluster", data=df2, fill=None, common_norm=False, palette="deep", zorder=1)
                df2["y"] = 0.1
                sns.scatterplot(data=df2, x="DA1", y="y", hue="Cluster", style="Original_Population", palette="deep")
                plt.legend(fontsize="small")
                plt.title("Un-RTLDA.v2 Embeddings (1 DA Axis)")

            pdf.savefig()
            plt.close()

def print_metrics(T, labels, G):
    """
    Calculate and print Adjusted Rand Index (ARI), Normalized Mutual Information (NMI),
    and Silhouette Score for the given embeddings, true labels, and predicted cluster labels.

    Args:
        T (numpy array): Un-RTLDA embeddings of shape (n_samples, n_components).
        labels (list): True labels for each sample.
        G (list): Predicted cluster assignments for each sample.

    Returns:
        None
    """
    ari = adjusted_rand_score(labels, G)
    nmi = normalized_mutual_info_score(labels, G)
    silhouette = silhouette_score(T, G)

    print("Adjusted Rand Index:", ari)
    print("Normalized Mutual Information:", nmi)
    print("Silhouette Score:", silhouette)

