import os 
import sys 

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from udapc.discriminant_analysis import unlda

# Simulate data
# X_sim, labels = simulate_genotypes()

# Generate synthetic data
n_samples = 1000
n_clusters = 4
n_features = 50
random_state = 1234

data, labels = unlda.generate_synthetic_data(n_samples=n_samples, 
				       n_clusters=n_clusters,
					    n_features=n_features, 
						random_state=random_state,
						dispersion=55.0)

# Apply Un-RTLDA and obtain the reduced-dimensional representation and cluster assignments
T, G, W, obj = unlda.un_rtlda(data, 
		   n_clusters, 
		   Ninit=10, 
		   max_iter=20, 
		   Ntry=10, 
		   center=True,
           gamma=0.001)

# Compute clustering performance metrics
unlda.print_metrics(T, labels, G)

# Call plot_embeddings on simulated data
unlda.plot_embeddings(T, G, W, data, labels, n_clusters)
