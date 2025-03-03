def set_seed_everywhere(seed, deterministic_cudnn=False):
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic_cudnn
    torch.backends.cudnn.benchmark = not deterministic_cudnn
