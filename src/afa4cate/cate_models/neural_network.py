"""Code adopted from the implementation of Causal-BALD by Jesson et al., available at https://github.com/anndvision/causal-bald."""

import torch
import numpy as np

from ignite import metrics

from torch import nn
from torch import optim
from torch.utils import data

from afa4cate.cate_models import core
from afa4cate.modules import dense
from afa4cate.modules import convolution
from afa4cate.modules import variational


class NeuralNetwork(core.PyTorchModel):
    def __init__(
        self,
        job_dir,
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
        mode='pi',
    ):
        super(NeuralNetwork, self).__init__(
            job_dir=job_dir,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            num_workers=num_workers,
            device=device,
        )
        encoder = (
            convolution.ResNet(
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
            if isinstance(dim_input, list)
            else dense.NeuralNetwork(
                architecture=architecture,
                dim_input=dim_input,
                dim_hidden=dim_hidden,
                depth=depth,
                negative_slope=negative_slope,
                batch_norm=batch_norm,
                dropout_rate=dropout_rate,
                spectral_norm=spectral_norm,
                activate_output=True,
            )
        )
        self.network = (
            nn.Sequential(
                encoder,
                variational.Categorical(
                    dim_input=encoder.dim_output,
                    dim_output=dim_output,
                ),
            )
            if mode == 'pi'
            else nn.Sequential(
                encoder,
                variational.Normal(
                    dim_input=encoder.dim_output,
                    dim_output=dim_output,
                )
            )
        )
        self.metrics = {
            "loss": metrics.Average(
                output_transform=lambda x: -x["outputs"].log_prob(x["targets"]).mean(),
                device=self.device,
            )
        }
        self.batch_size = batch_size
        self.best_loss = 1e7
        self.patience = patience
        self.optimizer = optim.Adam(
            params=self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=weight_decay,
        )
        self.network.to(self.device)

    def train_step(self, engine, batch):
        self.network.train()
        inputs, targets = self.preprocess(batch)
        self.optimizer.zero_grad()
        outputs = self.network(inputs)
        loss = -outputs.log_prob(targets).mean()
        loss.backward()
        self.optimizer.step()
        metric_values = {"outputs": outputs, "targets": targets}
        return metric_values

    def tune_step(self, engine, batch):
        self.network.eval()
        inputs, targets = self.preprocess(batch)
        with torch.no_grad():
            outputs = self.network(inputs)
        metric_values = {"outputs": outputs, "targets": targets}
        return metric_values

    def predict_mean(self, ds, batch_size=None):
        dl = data.DataLoader(
            ds,
            batch_size=2 * self.batch_size if batch_size is None else batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
        mean = []
        self.network.eval()
        with torch.no_grad():
            for batch in dl:
                batch = self.preprocess(batch)
                inputs, _ = batch
                posterior_predictive = self.network(inputs)
                mean.append(posterior_predictive.mean)
        return torch.cat(mean, 0).to("cpu").numpy()

    def predict_mean_from_covariates(self, X_samples, batch_size=None, seed=None):
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
            num_workers=self.num_workers,
            pin_memory=use_pin_memory
        )
        mean = []
        self.network.eval()
        with torch.no_grad():
            for covariates in dl:
                posterior_predictive = self.network(covariates)
                mean.append(posterior_predictive.mean)
        
        # Concatenate all batches into final tensors
        mean_pred = torch.cat(mean, 0).view(n_j, n_samples).to("cpu").numpy()
    
        return mean_pred
