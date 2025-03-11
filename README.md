# AFA4CATE: Active Feature Acquisition for Personalised Treatment Assignment

| **[Abstract](#abstract)**
| **[Installation](#installation)**
| **[Reproducing Experiments](#reproducing)**
| **[Citation](#citation)**

The code used to reproduce the numerical results presented in the paper *Active Feature Acquisition for Personalised Treatment Assignment*.

## Abstract

Making treatment effect estimation actionable for personalized decision-making requires overcoming the costs and delays of acquiring necessary features. While many machine learning models estimate Conditional Average Treatment Effects (CATE), they mostly assume that _all_ relevant features are readily available at prediction time -- a scenario that is rarely realistic. In practice, acquiring features, such as medical tests, can be both expensive and time-consuming, highlighting the need for strategies that select the most informative features for each individual, enhancing decision accuracy while controlling costs. Existing active feature acquisition (AFA) methods, developed for supervised learning, fail to address the unique challenges of CATE, such as confounding, overlap, and the structural similarities of potential outcomes under different treatments. To tackle these challenges, we propose specialised feature acquisition metrics and estimation strategies tailored to the CATE setting. We demonstrate the effectiveness of our methods through experiments on synthetic datasets designed to reflect common biases and data issues. In doing so, this work aims to bridge the gap between cutting-edge CATE estimation techniques and their practical, cost-efficient application in personalised treatment assignment.

## Installation

```.sh
$ git clone https://github.com/j-piskorz/afa4cate.git
$ cd afa4cate
$ conda create -n afa4cate
$ conda activate afa4cate
$ pip install -r requirements.txt
$ pip install .
```

## Reproducing Experiments

The results of the experiments are saved and processed using Weights&Biases. To be able to save the results of the experiments and then reproduce the figures presented in the paper, follow [these instructions](https://docs.wandb.ai/quickstart/#install-the-wandb-library-and-log-in) to sign up, create an API key and login into W&B.

By default, the experiments run on the CPU. To use a GPU instead, follow [these instructions](https://pytorch.org/get-started/locally/) to download the correct version of `pytorch` and then use the setting `device='cuda:0'` (or the corresponding GPU name) in the relevant bash file.

The results presented in Figure 2 can then be reproduced by running:
```.sh
cd experiments/
bash acic2016_acquisition_loop.sh
bash ihdp_acquisition_loop.sh
```
After all the code finished running, the results can be plotted by following the code in `experiments/notebooks/graphs_acquisition.ipynb`.

The results presented in Figure 3 can be reproduced by running:
```.sh
cd experiments/
bash ihdp_acquisition_stop.sh
```
After all the code finished running, the results can be plotted by following the code in `experiments/notebooks/graphs_acquisition_stop.ipynb`.

The hyperparameters learned obtained when running the experiments for the paper are saved in the `experiments/tuned_files` folder and will be used by default. To rerun hyperparameter tuning, delete these files.


