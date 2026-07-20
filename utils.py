import random
import numpy as np
import torch
import os


def set_seed(seed=0, full_deterministic=False):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if full_deterministic:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
            torch.use_deterministic_algorithms(True, warn_only=False)
            # Enable CuDNN deterministic mode
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_new_starting_seq(dataset):
    seqs = np.array(dataset.train.tolist() + dataset.valid.tolist())
    scores = np.array(dataset.train_scores.tolist() + dataset.valid_scores.tolist())
    return seqs[np.argmax(scores)]
