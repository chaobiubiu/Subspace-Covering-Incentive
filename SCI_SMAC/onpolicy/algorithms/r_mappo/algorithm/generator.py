import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from onpolicy.algorithms.utils.mlp import MLPLayer
from onpolicy.algorithms.utils.util import init
from onpolicy.utils.util import check, get_shape_from_obs_space


class MaskGenerator(nn.Module):
    def __init__(self, args, device=torch.device("cpu")):
        super(MaskGenerator, self).__init__()
        self.inp_size = args.num_objects
        self.hidden_size = args.hidden_size
        self.output_size = args.num_entities
        self.tpdv = dict(dtype=torch.float32, device=device)

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=np.sqrt(2))

        self.predictor = nn.Sequential(
            MLPLayer(self.inp_size,
                     self.hidden_size,
                     layer_N=args.layer_N-1,        # Only use 1fc+ReLU+fc
                     use_orthogonal=True,
                     use_ReLU=True,
                     use_LayerNorm=True),
            init_(nn.Linear(self.hidden_size, self.output_size * 2))
        )

        self.to(device)

    def forward(self):
        onehot_inps = check(np.eye(self.inp_size)).to(**self.tpdv)      # (num_objects, num_objects)
        entity_masks_logits = self.predictor(onehot_inps)          # (num_objects, num_entities * 2)
        # TODO: check the param hard.
        entity_masks = F.gumbel_softmax(entity_masks_logits.reshape(self.inp_size, self.output_size, 2), tau=0.01, hard=True, dim=-1)      # (num_objects, num_entities, 2)
        entity_masks = entity_masks[:, :, 1]        # (num_objects, num_entities)

        return entity_masks