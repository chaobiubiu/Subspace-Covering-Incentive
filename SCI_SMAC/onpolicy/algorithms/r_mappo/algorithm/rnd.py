import numpy as np
import torch
import torch.nn as nn
from onpolicy.algorithms.utils.mlp import MLPLayer
from onpolicy.algorithms.utils.util import init
from onpolicy.utils.util import check, get_shape_from_obs_space


class RND(nn.Module):
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(RND, self).__init__()
        self.hidden_size = args.hidden_size
        self.representation_size = args.rnd_rep_size
        self.tpdv = dict(dtype=torch.float32, device=device)

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=np.sqrt(2))
        
        self.cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        if len(self.cent_obs_shape) == 3:
            raise NotImplementedError
        else:
            self.predictor = nn.Sequential(
                MLPLayer(self.cent_obs_shape[0],
                         self.hidden_size, 
                         layer_N=args.layer_N + 2,      # 2 more layers than the target
                         use_orthogonal=True, 
                         use_ReLU=True,
                         use_LayerNorm=True),
                init_(nn.Linear(self.hidden_size, self.representation_size))
            )
            
            self.target = nn.Sequential(
                MLPLayer(self.cent_obs_shape[0],
                         self.hidden_size,
                         layer_N=args.layer_N,
                         use_orthogonal=True,
                         use_ReLU=True,
                         use_LayerNorm=True),
                init_(nn.Linear(self.hidden_size, self.representation_size))
            )

        for param in self.target.parameters():
            param.requires_grad = False
            
        self.to(device)

    def forward(self, cent_obs):
        cent_obs = check(cent_obs).to(**self.tpdv)
        predict_feature = self.predictor(cent_obs)
        target_feature = self.target(cent_obs)
        
        return predict_feature, target_feature