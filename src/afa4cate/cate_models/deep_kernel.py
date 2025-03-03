"""Code adopted from the implementation of Causal-BALD by Jesson et al., available at https://github.com/anndvision/causal-bald."""

import torch
import numpy as np

from torch import nn
from torch import optim
from torch.utils import data

from gpytorch import mlls
from gpytorch import likelihoods

from ignite import metrics

from afa4cate.cate_models import core
from afa4cate.modules import dense
from afa4cate.modules import convolution
from afa4cate.modules import gaussian_process


class DeepKernelGP(core.PyTorchModel):
    def __init__(
        self,
        job_dir,
        kernel,
        num_inducing_points,
        inducing_point_dataset,
        architecture,
        dim_input,
        dim_hidden,
        dim_output,
        depth,
        negative_slope,
        batch_norm,
        spectral_norm,
        dropout_rate,
        weight_decay,
        learning_rate,
        batch_size,
        epochs,
        patience,
        num_workers,
        seed,
        device=None,
    ):
        super(DeepKernelGP, self).__init__(
            job_dir=job_dir,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            num_workers=num_workers,
            device=device,
        )
        if isinstance(dim_input, list):
            self.encoder = convolution.ResNet(
                dim_input=dim_input,
                layers=[2] * depth,
                base_width=dim_hidden // 8,
                negative_slope=negative_slope,
                dropout_rate=dropout_rate,
                batch_norm=batch_norm,
                spectral_norm=spectral_norm,
                stem_kernel_size=5,
                stem_kernel_stride=1,
                stem_kernel_padding=2,
                stem_pool=False,
                activate_output=True,
            )
        else:
            self.encoder = nn.Sequential(
                dense.NeuralNetwork(
                    architecture=architecture,
                    dim_input=dim_input,
                    dim_hidden=dim_hidden,
                    depth=depth,
                    negative_slope=negative_slope,
                    batch_norm=batch_norm,
                    dropout_rate=dropout_rate,
                    spectral_norm=spectral_norm,
                    activate_output=False,
                ),
                dense.Activation(
                    dim_input=None,
                    negative_slope=negative_slope,
                    dropout_rate=0.0,
                    batch_norm=batch_norm,
                ),
            )

        self.encoder.to(self.device)
        self.dim_input = dim_input
        self.batch_size = batch_size
        self.best_loss = 1e7
        self.patience = patience
        (
            initial_inducing_points,
            initial_lengthscale,
        ) = gaussian_process.initial_values_for_GP(
            train_dataset=inducing_point_dataset,
            feature_extractor=self.encoder,
            n_inducing_points=num_inducing_points,
            device=self.device,
        )
        self.gp = gaussian_process.VariationalGP(
            num_outputs=dim_output,
            initial_lengthscale=initial_lengthscale,
            initial_inducing_points=initial_inducing_points,
            separate_inducing_points=False,
            kernel=kernel,
            ard=None,
            lengthscale_prior=False,
        ).to(self.device)
        self.network = gaussian_process.DeepKernelGP(
            encoder=self.encoder,
            gp=self.gp,
        )
        self.likelihood = likelihoods.GaussianLikelihood()
        self.optimizer = optim.Adam(
            params=[
                {"params": self.encoder.parameters(), "lr": self.learning_rate},
                {"params": self.gp.parameters(), "lr": 2 * self.learning_rate},
                {"params": self.likelihood.parameters(), "lr": 2 * self.learning_rate},
            ],
            weight_decay=weight_decay,
        )
        self.loss = mlls.VariationalELBO(
            likelihood=self.likelihood,
            model=self.network.gp,
            num_data=len(inducing_point_dataset),
        )
        self.metrics = {
            "loss": metrics.Average(
                output_transform=lambda x: -self.likelihood.expected_log_prob(
                    x["targets"].squeeze(), x["outputs"]
                ).mean(),
                device=self.device,
            )
        }
        self.network.to(self.device)
        self.likelihood.to(self.device)

    def train_step(self, engine, batch):
        self.network.train()
        self.likelihood.train()
        inputs, targets = self.preprocess(batch)
        self.optimizer.zero_grad()
        outputs = self.network(inputs)
        loss = -self.loss(outputs, targets.squeeze()).mean()
        loss.backward()
        self.optimizer.step()
        metric_values = {
            "outputs": outputs,
            "targets": targets,
        }
        return metric_values

    def tune_step(self, engine, batch):
        self.network.eval()
        self.likelihood.eval()
        inputs, targets = self.preprocess(batch)
        with torch.no_grad():
            outputs = self.network(inputs)
        metric_values = {
            "outputs": outputs,
            "targets": targets,
        }
        return metric_values

    def predict_mus(self, ds, posterior_sample=541, batch_size=None, seed=None):
        # Set the seed for reproducibility, if provided
        if seed is not None:
            torch.manual_seed(seed)  # For PyTorch random operations
            np.random.seed(seed)     # If NumPy random operations are used
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)  # Ensure reproducibility for CUDA

        dl = data.DataLoader(
            ds,
            batch_size=2 * self.batch_size if batch_size is None else batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        mu_0 = []
        mu_1 = []
        self.network.eval()
        with torch.no_grad():
            for batch in dl:
                batch = self.preprocess(batch)
                covariates = torch.cat([batch[0][:, :-1], batch[0][:, :-1]], 0)
                treatments = torch.cat(
                    [
                        torch.zeros_like(batch[0][:, -1:]),
                        torch.ones_like(batch[0][:, -1:]),
                    ],
                    0,
                )
                inputs = torch.cat([covariates, treatments], -1)
                posterior_predictive = self.network(inputs)
                samples = posterior_predictive.sample(torch.Size([posterior_sample]))
                mus = samples.chunk(2, dim=1)
                mu_0.append(mus[0])
                mu_1.append(mus[1])
        
        mu_0_pred = torch.cat(mu_0, 1).to("cpu").numpy().T
        mu_1_pred = torch.cat(mu_1, 1).to("cpu").numpy().T

        return mu_0_pred, mu_1_pred
        
    
    def predict_mus_from_covariates(self, X_samples, posterior_sample=50, mode='sample', batch_size=None, seed=None):
        """
        Function to process the covariates generated from conditional_normal_batch
        and return predictions from the posterior.

        Args:
        - X_samples: torch.Tensor of shape (n_j, n_samples, d), where n_j is the number of versions of X 
                     and n_samples is the number of posterior samples per version.
        - posterior_sample: int, number of posterior samples to draw from the network.
        - batch_size: int, batch size for processing. If None, default is used.

        Returns:
        - mu_0_predictions: numpy array of shape (n_j, n_samples, posterior_sample)
        - mu_1_predictions: numpy array of shape (n_j, n_samples, posterior_sample)
        """
        # Set the seed for reproducibility, if provided
        if seed is not None:
            torch.manual_seed(seed)  # For PyTorch random operations
            np.random.seed(seed)     # If NumPy random operations are used
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)  # Ensure reproducibility for CUDA

        # Get the device of the network
        device = next(self.network.parameters()).device

        # Move X_samples to the same device as the model
        X_samples = X_samples.to(device)

        n_j, n_samples, d = X_samples.shape

        # Reshape X_samples to (n_j * n_samples, d) for batch processing
        X_flat = X_samples.view(n_j * n_samples, d)

        # Determine if we should use pin_memory (only for CPU tensors)
        use_pin_memory = (device == torch.device('cpu'))

        # Create a DataLoader for the reshaped X_samples
        dl = data.DataLoader(
            X_flat, 
            batch_size=2 * self.batch_size if batch_size is None else batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=use_pin_memory
        )

        mu_0 = []
        mu_1 = []
        self.network.eval()  # Set the network in evaluation mode
        with torch.no_grad():
            for covariates in dl:
                # `covariates` is a batch of shape (batch_size, d), with no treatment column
                # Since no treatment info is present, we generate treatments manually
                treatments_zeros = torch.zeros_like(covariates[:, -1:], dtype=covariates.dtype)  # (batch_size, 1)
                treatments_ones = torch.ones_like(covariates[:, -1:], dtype=covariates.dtype)   # (batch_size, 1)

                # Duplicate covariates for both treatment arms
                covariates_repeated = torch.cat([covariates, covariates], dim=0)  # (2 * batch_size, d)
                treatments = torch.cat([treatments_zeros, treatments_ones], dim=0)  # (2 * batch_size, 1)

                # Concatenate covariates with treatments
                inputs = torch.cat([covariates_repeated, treatments], dim=-1)  # (2 * batch_size, d + 1)

                # Pass the inputs through the posterior predictive network
                posterior_predictive = self.network(inputs) # returns a MultivariateNormal object
                
                if mode == 'sample':
                    # Draw posterior samples
                    samples = posterior_predictive.sample(torch.Size([posterior_sample]))  # Shape: (posterior_sample, 2 * batch_size, ...)

                    # Split the samples into two parts: for treatment 0 and treatment 1
                    mus = samples.chunk(2, dim=1)  # (posterior_sample, batch_size, ...)

                    mu_0.append(mus[0])  # Collect predictions for treatment 0
                    mu_1.append(mus[1])  # Collect predictions for treatment 1
                
                elif mode == 'mean_var':
                    mu_mean = posterior_predictive.mean  # Shape: (2 * batch_size, ...)
                    mu_var = posterior_predictive.variance  # Shape: (2 * batch_size, ...)

                    # Split the mean and variance into two parts: for treatment 0 and treatment 1
                    mu_0_mean, mu_1_mean = mu_mean.chunk(2, dim=0)  # (batch_size, ...)
                    mu_0_var, mu_1_var = mu_var.chunk(2, dim=0)  # (batch_size, ...)

                    # Stack mean and variance to shape (batch_size, 2)
                    mu_0 = torch.stack([mu_0_mean, mu_0_var], dim=-1)  # (batch_size, 2)
                    mu_1 = torch.stack([mu_1_mean, mu_1_var], dim=-1)  # (batch_size, 2)

                    mu_0.append(mu_0)
                    mu_1.append(mu_1)
                
                else:
                    raise ValueError(f"Invalid mode: {mode}. Choose either 'sample' or 'mean_var'.")
                
        if mode == 'sample':
            # Concatenate all batches into final tensors
            mu_0_tensor = torch.cat(mu_0, dim=1).T  # Shape: (posterior_sample, n_j * n_samples)
            mu_1_tensor = torch.cat(mu_1, dim=1).T  # Shape: (posterior_sample, n_j * n_samples)

            # Reshape back to (n_j, n_samples, posterior_sample)
            mu_0_predictions = mu_0_tensor.view(n_j, n_samples, posterior_sample).to("cpu").numpy()
            mu_1_predictions = mu_1_tensor.view(n_j, n_samples, posterior_sample).to("cpu").numpy()

        elif mode == 'mean_var':
            # Concatenate all batches into final tensors
            mu_0_tensor = torch.cat(mu_0, dim=0)  # Shape: (n_j * n_samples, 2)
            mu_1_tensor = torch.cat(mu_1, dim=0)  # Shape: (n_j * n_samples, 2)

            # Reshape back to (n_j, n_samples, 2)
            mu_0_predictions = mu_0_tensor.view(n_j, n_samples, 2).to("cpu").numpy()
            mu_1_predictions = mu_1_tensor.view(n_j, n_samples, 2).to("cpu").numpy()
            
        return mu_0_predictions, mu_1_predictions

    
    def compute_pehe(self, ds, seed=None):
        mu_0, mu_1 = self.predict_mus(ds, seed=seed)
        # bring to the initial scale
        mu_0_avg = mu_0.mean(axis=1)
        mu_1_avg = mu_1.mean(axis=1)
        tau_hat = mu_1_avg - mu_0_avg
        tau = ds.tau

        mu_0_mse = np.sqrt(((mu_0_avg - ds.y0)**2).mean())
        mu_1_mse = np.sqrt(((mu_1_avg - ds.y1)**2).mean())
        
        y = self.predict_y(ds, seed=seed)
        y = y.mean(axis=1)
        y_mse = np.sqrt(((y - ds.y)**2).mean())

        pehe = np.sqrt(((tau_hat - tau) ** 2).mean())
        pehe_percentage = (np.abs((tau_hat - tau)/tau)).mean()
        return pehe, pehe_percentage, mu_0_mse, mu_1_mse, y_mse
    
    def predict_y(self, ds, posterior_sample=541, batch_size=None, seed=None):
        if seed is not None:
            torch.manual_seed(seed)  # For PyTorch random operations
            np.random.seed(seed)     # If NumPy random operations are used
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)  # Ensure reproducibility for CUDA

        dl = data.DataLoader(
            ds,
            batch_size=2 * self.batch_size if batch_size is None else batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        y = []
        self.network.eval()
        with torch.no_grad():
            for batch in dl:
                batch = self.preprocess(batch)
                inputs = batch[0]
                posterior_predictive = self.network(inputs)
                samples = posterior_predictive.sample(torch.Size([posterior_sample]))
                y.append(samples)
        
        y_pred = torch.cat(y, 1).to("cpu").numpy().T

        return y_pred

    def calculate_loss(self, ds, batch_size=None):
        dl = data.DataLoader(
            ds,
            batch_size=2 * self.batch_size if batch_size is None else batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        loss = []
        self.network.eval()
        with torch.no_grad():
            for batch in dl:
                batch = self.preprocess(batch)
                inputs, targets = batch
                outputs = self.network(inputs)
                loss.append(-self.likelihood.expected_log_prob(targets.squeeze(), outputs).to("cpu").numpy().mean())
        loss = np.array(loss).mean()
        return loss
