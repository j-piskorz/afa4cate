import numpy as np
import torch
from torch.utils import data

def alpha_fn(pi, lambda_):
    return (pi * lambda_) ** -1 + 1.0 - lambda_ ** -1

def beta_fn(pi, lambda_):
    return lambda_ * (pi) ** -1 + 1.0 - lambda_

def assign_treatment(X, setup, rng):
    d = X.shape[1]
    if setup == 'random':
        return np.ones(X.shape[0])*0.5
    elif setup == 'confounding':
        beta_j = rng.binomial(1, 0.4, size=d).astype('float32')
        linear_terms = np.dot(X, beta_j)
        range_vals = np.max(linear_terms) - np.min(linear_terms)
        linear_terms = (linear_terms/(range_vals/2)*2.2)
        return 1/(1+np.exp(-linear_terms)), None
    elif setup == 'confounding_imbalance':
        beta_j = rng.binomial(1, 0.4, size=d).astype('float32')
        linear_terms = np.dot(X, beta_j)
        range_vals = np.max(linear_terms) - np.min(linear_terms)
        linear_terms = (linear_terms/(range_vals/2) - 0.5)
        linear_terms[linear_terms < 0] *= 1.4
        linear_terms[linear_terms > 0] *= 4.4
        return 1/(1+np.exp(-linear_terms)), None
    elif setup == 'overlap_violation':
        overlap_vars = rng.choice(d, size=4, replace=False)
        beta_j = rng.binomial(1, 0.4, size=d).astype('float32')
        linear_terms = np.dot(X, beta_j)
        range_vals = np.max(linear_terms) - np.min(linear_terms)
        linear_terms = (linear_terms/(range_vals/2)*2.2)
        prob = 1/(1+np.exp(-linear_terms))
        # violate overlap
        for i in range(X.shape[0]):
            for j in overlap_vars[:2]:
                if X[i, j] > np.quantile(X[:, j], 0.90):
                    prob[i] = 1.0
            for j in overlap_vars[2:]:
                if X[i, j] < np.quantile(X[:, j], 0.10):
                    prob[i] = 0.0
        return prob, overlap_vars
    else:
        raise NotImplementedError(f"Treatment assignment according to {setup} not supported. Choose 'random', 'confounding', or 'confounding_imbalance'")
    
def complete_propensity(x, u, lambda_, beta=0.75):
    nominal = nominal_propensity(x, beta=beta)
    alpha = alpha_fn(nominal, lambda_)
    beta = beta_fn(nominal, lambda_)
    return (u / alpha) + ((1 - u) / beta)

def nominal_propensity(x, beta=0.75):
    logit = beta * x + 0.5
    return (1 + np.exp(-logit)) ** -1

def f_mu(x, t, u, gamma=4.0):
    mu = (
        (2 * t - 1) * x
        + (2.0 * t - 1)
        - 2 * np.sin((4 * t - 2) * x)
        - (gamma * u - 2) * (1 + 0.5 * x)
    )
    return mu

def linear_normalization(x, new_min, new_max):
    return (x - x.min()) * (new_max - new_min) / (x.max() - x.min()) + new_min

def lambda_top_func(mu, k, y, alpha):
    m = y.shape[0]
    r = (y[k:] - mu).sum(dim=0)
    py = (k + 1) / m
    return mu + r.div(m * (alpha + 1) - py)


def lambda_bottom_func(mu, k, y, alpha):
    m = y.shape[0]
    r = (y[: k + 1] - mu).sum(dim=0)
    py = k + 1
    return mu + r.div(m * alpha + py)


class RandomFixedLengthSampler(data.Sampler):
    """
    Sometimes, you really want to do more with little data without increasing the number of epochs.
    This sampler takes a `dataset` and draws `target_length` samples from it (with repetition).
    """

    def __init__(self, dataset, target_length):
        super().__init__(dataset)
        self.dataset = dataset
        self.target_length = target_length

    def __iter__(self):
        # Ensure that we don't lose data by accident.
        if self.target_length < len(self.dataset):
            return iter(torch.randperm(len(self.dataset)).tolist())

        # Sample slightly more indices to avoid biasing towards start of dataset
        indices = torch.randperm(
            self.target_length + (-self.target_length % len(self.dataset))
        )

        return iter((indices[: self.target_length] % len(self.dataset)).tolist())

    def __len__(self):
        return self.target_length
