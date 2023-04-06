import os 
import sys 

import random
import numpy as np
import seaborn as sns
import pandas as pd

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


def generate_synthetic_data(n_samples=1000, n_clusters=4, n_features=50, random_state=None, dispersion=1):
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
	eigvals, eigvecs = eigh(A, B)
	ind = np.argsort(eigvals)
	eigvals_sorted = eigvals[ind]
	eigvecs_sorted = eigvecs[:, ind]
	
	model = {'W': eigvecs_sorted, 'D': np.diag(eigvals_sorted)}
	return model

# Define the Un-RTLDA function
def un_rtlda(X, c, Ninit=10, gamma=1e-6, tol=1e-6, max_iter=100, Ntry=10, center=True, no_pca=False):
	n, d = X.shape  # Number of samples
	
	if center:
		H = np.eye(n) - np.ones((n, n)) / n 
	else:
		H = np.eye(n)

	St = X.T @ H @ X  # Compute the within-class scatter matrix St

	# print("X:",X.shape)
	# print("H:",H.shape)
	# print("St:",St.shape)

	Stt = St + gamma * np.eye(d)  # Add regularization term to St
	#print("Stt:",Stt.shape)

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
		W = W = X.T[:, :m]
	else:
		W = pca.fit_transform(X.T @ H)
	W2 = W  # Initialize W2 with W

	#print("W:",W.shape)

	# Iterate until convergence or maxIter is reached
	#while (not np.isclose(obj_old, obj_new, atol=tol) or it == 0) and it < max_iter:
	while (obj_new - obj_old) > tol and it < max_iter:
	#while it < max_iter:

		it += 1
		obj_old = obj_new

		# print("W2:",W.shape)
		# print("W2.T:",W.T.shape)
		# print("Stt:",Stt.shape)
		# print("X:",X.shape)
		# print("X.T:",X.T.shape)
		# print("H:",H.shape)
		# print("H.T:",H.T.shape)
		# Calculate the intermediate matrix product
		T = (fractional_matrix_power(W2.T @ Stt @ W2, -0.5) @ W2.T @ X.T @ H).T

		# Check the dimensions of M
		#print("T.shape:", T.shape)  # Should output (2, 100)

		best_obj_tmp = float('inf')
		best_Ypre = None

		# Loop through Ntry times to find the best clustering
		for j in range(Ntry+1):
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
		#print("Sb:",Sb.shape)


		# Perform generalized eigenvalue decomposition and update W2
		eigvals, eigvecs = eigh(Sb, Stt)
		W2 = eigvecs[:, -m:]

		# Update the new objective value
		obj_new = np.trace((W2.T @ Stt @ W2) ** -1 @ W2.T @ Sb @ W2)
		print(obj_new)

	# Print a warning if the algorithm did not converge within maxIter iterations
	if it == max_iter:
		print(f"Warning: The un_rtlda did not converge within {max_iter} iterations!")

	return T, Ypre, W2

def plot_embeddings(T, G, W, dataset, labels, n_clusters, filename="embeddings_plots.pdf"):
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
			plt.figure()
			sns.scatterplot(data=df2, x="DA1", y="DA2", hue="Cluster", style="Original_Population", palette="deep")
			plt.title("Un-RTLDA.v2 Embeddings")
			pdf.savefig()
			plt.close()

def simulate_genotypes(n=100, d=1000):
	p1, p2, p3 = np.random.uniform(0.05, 0.95, size=3)
	q1, q2, q3 = 1 - p1, 1 - p2, 1 - p3

	P = np.array([[q1*q1, 2*p1*q1, p1*p1], 
				  [q2*q2, 2*p2*q2, p2*p2], 
				  [q3*q3, 2*p3*q3, p3*p3]])

	pop_sizes = [int(0.4*n), int(0.3*n), int(0.3*n)]
	labels = np.concatenate((np.zeros(pop_sizes[0]), np.ones(pop_sizes[1]), 2*np.ones(pop_sizes[2])))
	X_sim = np.zeros((n, d), dtype=np.int8)
	for i in range(3):
		mask = labels == i
		X_sim[mask] = np.random.choice([0, 1, 2], size=(np.sum(mask), d), p=P[i])
	
	return X_sim, labels

def print_metrics(T, labels, G):
	ari = adjusted_rand_score(labels, G)
	nmi = normalized_mutual_info_score(labels, G)
	silhouette = silhouette_score(T, G)

	print("Adjusted Rand Index:", ari)
	print("Normalized Mutual Information:", nmi)
	print("Silhouette Score:", silhouette)

