import numpy as np
from scipy.optimize import minimize

np.random.seed(0)
true_mu = np.array([5.0, 3.0, 1.0])
true_cov = np.array([[2.0, 0.5, 0.3],
                     [0.5, 1.0, 0.2],
                     [0.3, 0.2, 1.5]])
data = np.random.multivariate_normal(true_mu, true_cov, 100)


def log_likelihood(params, data):
    mu = params[:3]
    cov_matrix = np.array([[params[3], params[4], params[5]],
                           [params[4], params[6], params[7]],
                           [params[5], params[7], params[8]]])

    epsilon = 1e-4
    cov_matrix += epsilon * np.eye(3)

    n = data.shape[0]
    d = data.shape[1]

    diff = data - mu
    try:
        log_det_cov = np.linalg.slogdet(cov_matrix)[1]
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        return np.inf

    # likelihood = -0.5 * (n * d * np.log(2 * np.pi) + n * log_det_cov + np.sum(diff @ inv_cov_matrix * diff))
    # print("diff   ", diff.shape)
    # print("inv    ", inv_cov_matrix.shape)
    temp = np.matmul(diff, inv_cov_matrix)
    temp = np.matmul(temp, diff.T)
    temp = np.trace(temp)
    likelihood = -0.5 * (n * d * np.log(2 * np.pi) + n * log_det_cov + np.sum(temp))

    return -likelihood

# def my_logpdf(params, x):
#     u = params[:3]  # The first three parameters are the mean vector
#     covar = np.array([[params[3], params[4], params[5]],
#                            [params[4], params[6], params[7]],
#                            [params[5], params[7], params[8]]])
#     k = len(x)  # dimension
#     a = np.transpose(x - u)
#     b = np.linalg.inv(covar)
#     c = x - u
#     d = np.matmul(a, b)
#     e = np.matmul(d, c)
#     numer = np.exp(-0.5 * e)
#     f = (2 * np.pi)**k
#     g = np.linalg.det(covar)
#     denom = np.sqrt(f * g)
#     pdf = numer / denom
#     return np.log(pdf)

initial_guess = np.array([5.0, 3.0, 1.0, 2.0, 0.5, 0.3, 1.0, 0.2, 1.5, 2.0, 1.0, 1.5])
# initial_guess2 = np.array([5.0, 3.0, 1.0, 2.0, 0.5, 0.3, 1.0, 0.2, 1.5, 2.0, 1.0, 1.5])

result = minimize(log_likelihood, initial_guess, args=(data,), method='L-BFGS-B')

# second_result = minimize(my_logpdf, initial_guess2, args=(data,), method='L-BFGS-B')
# print(second_result)
estimated_mu = result.x[:3]
estimated_cov = np.array([[result.x[3], result.x[4], result.x[5]],
                          [result.x[4], result.x[6], result.x[7]],
                          [result.x[5], result.x[7], result.x[8]]])

# Display the MLE results
# print("original mu   ", np.mean(data, axis=0))
print("Estimated mu: ", estimated_mu)
print("Estimated covariance matrix:\n", estimated_cov)
