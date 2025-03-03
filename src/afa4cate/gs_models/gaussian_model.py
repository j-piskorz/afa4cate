import torch
import numpy as np
from pathlib import Path
from afa4cate.datasets.load_acic2016 import load
from ignite import distributed

class GaussianGSM:
    def __init__(self, d, rho_cov, device=None):
        """
        Initialize the Gaussian Structural Generative Model.
        
        Args:
        - mu: torch.Tensor of shape (d, ) representing the mean vector of the multivariate normal distribution.
        - Sigma: torch.Tensor of shape (d, d) representing the covariance matrix of the multivariate normal distribution.
        """
        if device is None:
            self.device = distributed.device()
        else:
            self.device = torch.device(device)

        if isinstance(rho_cov, str) and rho_cov == 'acic2016':
            # Load the mean and covariance from the ACIC 2016 dataset
            path = Path(__file__).parent.parent / "datasets" / "acic2016"
            path.mkdir(parents=True, exist_ok=True)
            acic = load(path, preprocessed=True, original_acic_outcomes=True)

            # Remove binary variables
            binary = [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                           19, 26, 27, 28, 29, 34, 35, 36, 37, 38, 42,
                           43, 44, 45, 46, 47, 48, 49, 50, 51, 52]
            acic = np.delete(acic, binary, axis=1)
            
            self.mu = torch.tensor(np.mean(acic, axis=0), dtype=torch.float32).to(self.device)
            self.Sigma = torch.tensor(np.cov(acic, rowvar=False), dtype=torch.float32).to(self.device)
        else:
            self.mu = torch.zeros(d).to(self.device)  # Mean vector
            self.Sigma = (rho_cov * torch.ones((d, d)) + (1 - rho_cov) * torch.eye(d)).to(self.device)  # Covariance matrix
    
    def define_mean_sigma(self, mu, Sigma):
        """
        Define the mean vector and covariance matrix of the multivariate normal distribution.
        
        Args:
        - mu: torch.Tensor of shape (d, ) representing the mean vector.
        - Sigma: torch.Tensor of shape (d, d) representing the covariance matrix.
        """
        self.mu = torch.tensor(mu, dtype=torch.float32).to(self.device)
        self.Sigma = torch.tensor(Sigma, dtype=torch.float32).to(self.device)

    def sample_conditional_covariates(self, X_batch, Z, n_samples=1, seed=None):
        """
        Sample from p(X | X_O) for a batch of n_j versions of X given observed parts X_O and the masking vector Z.
        
        Args:
        - X_batch: torch.Tensor of shape (n_j, d) representing the batch of n_j versions of X.
        - Z: torch.Tensor of shape (d, ) representing the masking vector, where 1 means observed and 0 means not observed.
        - n_samples: int, number of samples to draw from p(X | X_O) for each X in the batch.
        - seed: int or None, random seed for reproducibility.
        
        Returns:
        - X_cond_samples: torch.Tensor of shape (n_j, n_samples, d) representing samples from p(X | X_O) for each X in the batch.
        """
        if seed is not None:
            torch.manual_seed(seed)  # Set the seed for reproducibility

        X_batch, Z = X_batch.to(self.device), Z.to(self.device)
        n_j, d = X_batch.shape

        # Identify the observed and unobserved indices
        observed_idx = Z.bool()  # Boolean mask for observed variables
        unobserved_idx = ~observed_idx
        observed_idx = observed_idx.to(self.device)
        unobserved_idx = unobserved_idx.to(self.device)

        # Split the mean vector
        mu_O = self.mu[observed_idx]  # Mean for observed variables
        mu_U = self.mu[unobserved_idx]  # Mean for unobserved variables

        # Split the covariance matrix
        Sigma_OO = self.Sigma[observed_idx][:, observed_idx]  # Covariance of observed variables
        Sigma_UO = self.Sigma[unobserved_idx][:, observed_idx]  # Covariance between unobserved and observed
        Sigma_UU = self.Sigma[unobserved_idx][:, unobserved_idx]  # Covariance of unobserved variables

        # Compute the inverse of Sigma_OO
        Sigma_OO_inv = torch.inverse(Sigma_OO)

        # Conditional covariance for unobserved variables
        conditional_cov = Sigma_UU - Sigma_UO @ Sigma_OO_inv @ Sigma_UO.T

        # Initialize a tensor to store the conditional samples for the whole batch
        X_cond_samples = torch.zeros((n_j, n_samples, d)).to(self.device)  # Shape: (n_j, n_samples, d)

        # Process each batch entry
        for j in range(n_j):
            X = X_batch[j]
            # Get the observed values for the current sample
            X_O = X[observed_idx]

            # Compute the conditional mean for unobserved variables for the current sample
            conditional_mean = mu_U + Sigma_UO @ Sigma_OO_inv @ (X_O - mu_O)

            # Vectorized sampling: generate all n_samples at once for the current sample
            dist = torch.distributions.MultivariateNormal(conditional_mean, conditional_cov)
            X_U_cond_samples = dist.sample((n_samples,))  # Shape: (n_samples, len(unobserved_idx))

            # Replicate the observed part of X across all samples
            X_cond_samples[j, :, :] = X.unsqueeze(0).repeat(n_samples, 1)  # Shape: (n_samples, d)

            # Replace the unobserved variables with the sampled values
            X_cond_samples[j, :, unobserved_idx] = X_U_cond_samples

        return X_cond_samples

    def sample_single_covariate(self, X, Z, j, n_samples=1, seed=None):
        """
        Sample the j-th unobserved covariate X_j from p(X_j | X_O), given observed parts X_O and the masking vector Z.
        
        Args:
        - X: numpy array of shape (d, ) representing the full vector X.
        - Z: numpy array of shape (d, ) representing the masking vector, where 1 means observed and 0 means unobserved.
        - mu: numpy array of shape (d, ) representing the mean vector of the multivariate normal distribution.
        - Sigma: numpy array of shape (d, d) representing the covariance matrix.
        - j: int, index of the covariate X_j to sample.
        - n_samples: int, number of samples to draw from p(X_j | X_O).
        - seed: int or None, random seed for reproducibility.
        
        Returns:
        - X_samples: torch.Tensor of shape (n_samples, d), where the j-th entry is filled with the sampled values.
        - Z_new: torch.Tensor of shape (d, ) where the j-th entry is updated to 1 (indicating X_j is now observed).
        """
        if seed is not None:
            torch.manual_seed(seed)  # Set the seed for reproducibility

        # Convert X, Z, mu, and Sigma to PyTorch tensors
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        Z = torch.tensor(Z, dtype=torch.float32).to(self.device)

        # Identify the observed and unobserved indices
        observed_idx = Z.bool()  # Boolean mask for observed variables
        observed_idx[j] = False  # We want to predict X_j, so treat it as unobserved in this case
        observed_idx = observed_idx.to(self.device)

        # Split the mean vector
        mu_O = self.mu[observed_idx]  # Mean for observed variables
        mu_j = self.mu[j]  # Mean for the j-th variable (we will sample this)

        # Split the covariance matrix
        Sigma_OO = self.Sigma[observed_idx][:, observed_idx]  # Covariance of observed variables
        Sigma_jO = self.Sigma[j, observed_idx]  # Covariance between the j-th variable and the observed variables
        Sigma_jj = self.Sigma[j, j]  # Variance of the j-th variable

        # Compute the inverse of Sigma_OO
        Sigma_OO_inv = torch.inverse(Sigma_OO)

        # Get the observed values
        X_O = X[observed_idx]

        # Compute the conditional mean and variance for X_j given X_O
        conditional_mean_j = mu_j + Sigma_jO @ Sigma_OO_inv @ (X_O - mu_O)
        conditional_var_j = Sigma_jj - Sigma_jO @ Sigma_OO_inv @ Sigma_jO.T

        # Generate n_samples for X_j from the normal distribution with the computed mean and variance
        X_j_samples = torch.normal(conditional_mean_j, torch.sqrt(conditional_var_j), size=(n_samples,))

        # Create a matrix to store the samples with updated X
        X_samples = X.unsqueeze(0).repeat(n_samples, 1)  # Shape: (n_samples, d)
        
        # Fill the j-th entry with the sampled values
        X_samples[:, j] = X_j_samples

        # Update the masking vector Z to reflect that X_j is now observed
        Z_new = Z.clone()
        Z_new[j] = 1

        return X_samples, Z_new
