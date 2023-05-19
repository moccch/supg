import numpy as np
from scipy.stats import bernoulli

# Set the seed for reproducibility
np.random.seed(0)

# Define the alpha and beta parameters for the Beta distributions
alpha_beta_pairs = (0.01, 1)
alpha = alpha_beta_pairs[0]
beta = alpha_beta_pairs[1]

# Define the number of records
n_records = 10**6

# A(x) - Proxy model generating scores from Beta distribution
def A(alpha, beta, n):
    return np.random.beta(alpha, beta, n)

# O(x) - Oracle model generating labels from Bernoulli distribution
def O(scores):
    return bernoulli.rvs(scores)

# Generate proxy scores and oracle labels for each pair of alpha and beta parameters

proxy_scores = A(alpha, beta, n_records)
oracle_labels = O(proxy_scores)

# Save the data
np.save('proxy_scores_dataset.npy', proxy_scores)
np.save('oracle_labels_dataset.npy', oracle_labels)
