import numpy as np
from scipy.optimize import minimize


np.random.seed(0)
true_mu = 0.0
true_sigma = 1.0
data = np.random.normal(true_mu, true_sigma, 10)

def log_likelihood(params, data):
    mu, sigma = params
    n = len(data)
    log_likelihood = -n/2 * np.log(2 * np.pi * sigma**2) - np.sum((data - mu)**2) / (2 * sigma**2)
    return -log_likelihood

initial_guess = [2, 2]

result = minimize(log_likelihood, initial_guess, args=(data,), method='L-BFGS-B', bounds=[(None, None), (1e-6, None)])

estimated_mu, estimated_sigma = result.x
estimated_data = np.random.normal(estimated_mu, estimated_sigma, 10)
co_data = np.vstack((data, estimated_data))

print(f"Estimated mu: {estimated_mu}")
print(f"Estimated sigma: {estimated_sigma}")
