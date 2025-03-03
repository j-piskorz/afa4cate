import numpy as np

def r_PO(mu0, mu1, pi=None):
    var_y1 = np.mean(mu1.var(axis=2).mean(axis=1) + mu1.mean(axis=2).var(axis=1))
    var_y0 = np.mean(mu0.var(axis=2).mean(axis=1) + mu0.mean(axis=2).var(axis=1))
    return -var_y1 - var_y0

def r_TE(mu0, mu1, pi=None):
    var_TE = np.mean((mu1 - mu0).var(axis=2).mean(axis=1) + (mu1 - mu0).mean(axis=2).var(axis=1))
    return -var_TE

def r_sTE(mu0, mu1, pi=None):
    var_sTE = np.mean((mu1 - mu0 > 0).var(axis=2).mean(axis=1) + (mu1 - mu0 > 0).mean(axis=2).var(axis=1))
    return -var_sTE


ACQUISITION_METRICS = {
    "r_PO": r_PO,
    "r_TE": r_TE,
    "r_sTE": r_sTE,
    "random": None,
}