import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils import data
from afa4cate.datasets import utils
from afa4cate.datasets.load_acic2016 import load

from sklearn import preprocessing
from sklearn import model_selection


class SyntheticCATEDataset(data.Dataset):
    def __init__(
        self,
        n_samples,
        setup_pi='random',  # 'random' for random treatment assignment
        sigma_y=1.0,        # Noise in outcome generation
        rho_TE=0.5,         # probability of non-zero treatment effect
        lambd=1.0,          # treatment effect strength
        seed=1331,          # Random seed
        mode='mu',          # 'pi' for propensity, 'mu' for outcome models
        split='train'       # 'train', 'valid', or 'test'
    ):
        super(SyntheticCATEDataset, self).__init__()
        rng = np.random.RandomState(seed=seed)
        self.n_samples = n_samples
        self.dim_treatment = 1
        self.dim_output = 1

        self.setup_pi = setup_pi

        # Step 1: Generate Covariates (multivariate normal with correlations)
        self.acic = True
        path = Path(__file__).parent / "acic2016"
        path.mkdir(parents=True, exist_ok=True)
        acic = load(path, preprocessed=True, original_acic_outcomes=True)

        # Remove binary variables
        binary = [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                        19, 26, 27, 28, 29, 34, 35, 36, 37, 38, 42,
                        43, 44, 45, 46, 47, 48, 49, 50, 51, 52]
        acic = np.delete(acic, binary, axis=1)
        
        self.mean = np.mean(acic, axis=0)
        self.cov = np.cov(acic, rowvar=False)
        self.d = acic.shape[1]
        self.dim_input = self.d
        x = rng.multivariate_normal(self.mean, self.cov, size=n_samples).astype('float32')

        # Step 2: Assign treatments
        pi, overlap = utils.assign_treatment(X=x, setup=setup_pi, rng=rng)
        pi = pi.astype("float32").ravel()
        t = rng.binomial(1, pi).astype("float32")
        self.overlap = overlap
        
        # Step 3: Generate potential outcomes based on the specified setup
        y0, y1, tau = self.generate_potential_outcomes_A(x, rng, rho_TE, lambd, self.acic, overlap)

        # Factual outcome (depending on treatment)
        y = t * y1 + (1 - t) * y0 + (sigma_y * rng.standard_normal(n_samples).astype('float32'))

        # create a dataframe
        xs = [f"x{i}" for i in range(self.d)]
        df_x = pd.DataFrame({xs[i]: x[:, i] for i in range(self.d)})
        df_else = pd.DataFrame({"y0": y0, "y1": y1, "tau": tau, "y": y, "t": t, "pi": pi})
        df = pd.concat([df_x, df_else], axis=1)
        
        # Train test split
        df_train, df_test = model_selection.train_test_split(
            df, test_size=0.1, random_state=seed
        )
        self.mode = mode
        self.split = split

        # Set x, y, and t values
        self.y_mean = (
            df_train["y"].to_numpy(dtype="float32").mean(keepdims=True)
            if mode == "mu"
            else np.asarray([0.0], dtype="float32")
        )
        self.y_std = (
            df_train["y"].to_numpy(dtype="float32").std(keepdims=True)
            if mode == "mu"
            else np.asarray([1.0], dtype="float32")
        )

        # standardising the outcomes
        df_train["y"] = (df_train["y"] - self.y_mean) / self.y_std
        df_test["y"] = (df_test["y"] - self.y_mean) / self.y_std
        df_train["y0"] = (df_train["y0"] - self.y_mean) / self.y_std
        df_test["y0"] = (df_test["y0"] - self.y_mean) / self.y_std
        df_train["y1"] = (df_train["y1"] - self.y_mean) / self.y_std
        df_test["y1"] = (df_test["y1"] - self.y_mean) / self.y_std
        df_train["tau"] = df_train["y1"] - df_train["y0"]
        df_test["tau"] = df_test["y1"] - df_test["y0"]
        
        if self.split == "test":
            self.x = df_test[xs].to_numpy(dtype="float32")
            self.t = df_test["t"].to_numpy(dtype="float32")
            self.y0 = df_test["y0"].to_numpy(dtype="float32")
            self.y1 = df_test["y1"].to_numpy(dtype="float32")
            self.y = df_test["y"].to_numpy(dtype="float32")
            self.tau = df_test["tau"].to_numpy(dtype="float32")
            self.pi = df_test["pi"].to_numpy(dtype="float32")
        else:
            df_train, df_valid = model_selection.train_test_split(
                df_train, test_size=0.3, random_state=seed
            )
            if split == "train":
                df = df_train
            elif split == "valid":
                df = df_valid
            else:
                raise NotImplementedError("Not a valid dataset split")
            self.x = df[xs].to_numpy(dtype="float32")
            self.t = df["t"].to_numpy(dtype="float32")
            self.y0 = df["y0"].to_numpy(dtype="float32")
            self.y1 = df["y1"].to_numpy(dtype="float32")
            self.y = df["y"].to_numpy(dtype="float32")
            self.tau = df["tau"].to_numpy(dtype="float32")
            self.pi = df["pi"].to_numpy(dtype="float32")
            
        # Set the mode for inputs and targets
        if mode == "pi":
            self.inputs = self.x
            self.targets = self.t
        elif mode == "mu":
            self.inputs = np.hstack([self.x, np.expand_dims(self.t, -1)])
            self.targets = self.y
        elif mode == 'var':
            self.inputs = self.x
            self.targets = self.variance
        else:
            raise NotImplementedError(f"{mode} not supported. Choose 'pi' for propensity or 'mu' for outcome models")

        # Convert inputs and targets to tensors
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def generate_potential_outcomes_A(self, X, rng, rho_TE, lambd, acic=False, overlap=None):
        """
        Generate potential outcomes based on Setup A.
        """
        n, d = X.shape

        # if acic:
        #     # binarise some of the covariates
        #     binary_vars = [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        #                    19, 26, 27, 28, 29, 34, 35, 36, 37, 38, 42,
        #                    43, 44, 45, 46, 47, 48, 49, 50, 51, 52]
        #     X[:, binary_vars] = (X[:, binary_vars] > 0.5).astype("float32")

        # generate parameters
        beta_j = rng.binomial(1, 0.6, size=d).astype('float32')
        gamma_j = rng.binomial(1, rho_TE, size=d).astype('float32')
        beta_jl = rng.binomial(1, 0.3, size=(d, d)).astype('float32')
        c = 1.0  # Intercept
        if self.setup_pi == 'overlap_violation':
            gamma_j[overlap] = 0
            beta_j[overlap] = 0
        self.predictive = np.where(gamma_j == 1)[0]

        # Calculate terms
        linear_terms = np.dot(X, beta_j)
        interaction_terms = np.sum(np.dot(X, beta_jl) * X, axis=1)
        treatment_terms = np.dot(X, gamma_j)

        # Potential outcomes
        y0 = c + linear_terms + interaction_terms
        y1 = y0 + lambd*treatment_terms  # For simplicity, we add treatment effect to y0
        tau = y1 - y0
        return y0, y1, tau


    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index : index + 1]
    
    def output_variance(self, variance):
        self.variance = variance
        self.mode = 'var'
