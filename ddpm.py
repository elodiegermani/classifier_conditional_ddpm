from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from model import ContextUnet
import importlib
import sys 
sys.path.insert(0, '../pipeline_classification/src')


def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, config):
        super(DDPM, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nn_model = ContextUnet(in_channels=1, n_feat=config.n_feat, n_classes=config.n_classes)
        self.betas = config.beta 
        self.n_T = config.n_T
        self.drop_prob = config.drop_prob
        self.n_classes = config.n_classes

        self.nn_model.to(self.device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(self.betas[0], self.betas[1], self.n_T).items():
            self.register_buffer(k, v)

        self.loss_mse = nn.MSELoss()

        sys.path.insert(0, '../pipeline_classification/src')
        package = 'lib.model'
        md = importlib.import_module(package)

        classifier = torch.load(config.model_param, map_location=self.device)
        classifier = classifier.state_dict()

        # Initialize the model with the pre-trained weights
        regressor = md.Regressor3D()
        regressor_dict = regressor.state_dict()

        # 1. filter out unnecessary keys
        classifier = {k: v for k, v in classifier.items() if k in regressor_dict}
        # 2. overwrite entries in the existing state dict
        regressor_dict.update(classifier) 
        # 3. load the new state dict
        regressor.load_state_dict(regressor_dict)

        self.classembed = regressor.to(self.device) 

    def forward(self, x):
        """
        this method is used in training, so samples t and noise randomly
        """
        # for sampling noise and real 
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab.to(self.device)[_ts, None, None, None, None] * x
            + self.sqrtmab.to(self.device)[_ts, None, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        cemb = self.classembed(x)
        
        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, _ts / self.n_T, cemb))

    def transfer(self, source, target):

        x_i = source.to(self.device)  # 
        noise = torch.randn_like(x_i)  # eps ~ N(0, 1)
        x_t = (
            self.sqrtab.to(self.device)[self.n_T] * x_i
            + self.sqrtmab.to(self.device)[self.n_T] * noise
        )

        cemb = self.classembed(target.to(self.device))

        for i in range(self.n_T, 0, -1):

            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(self.device)
            t_is = t_is.repeat(1,1,1,1,1)

            z = torch.randn(*x_t.shape).to(self.device) if i > 1 else 0

            # split predictions and compute weighting  
            eps = self.nn_model(x_t.float(), t_is.float(), cemb.float())
            x_t = (
                self.oneover_sqrta[i] * (x_t - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            ) 
        
        return x_t