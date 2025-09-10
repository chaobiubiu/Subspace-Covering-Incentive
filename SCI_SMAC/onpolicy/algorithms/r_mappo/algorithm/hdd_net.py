import numpy as np
import torch
import torch.nn as nn
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.util import check
from onpolicy.utils.util import get_grad_norm, get_shape_from_obs_space


class HDD(nn.Module):
    def __init__(self, act_space, obs_space, cent_obs_space, args, device=torch.device("cpu")):
        super(HDD, self).__init__()
        self.hidden_size = args.hidden_size
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        obs_shape = get_shape_from_obs_space(obs_space)
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)

        if len(obs_shape) == 3: # image input
            raise NotImplementedError

        obs_dim = obs_shape[0]
        cent_obs_dim = cent_obs_shape[0]

        input_size = obs_dim + cent_obs_dim     # o_{t}^{i}+s_{t+1}^{j} as inputs
        
        self.base = MLPBase(args, (input_size,))   
        self.act = ACTLayer(act_space, self.hidden_size, args.use_orthogonal, args.gain)

        self.to(device)

    def forward(self, obs, cent_obs, available_actions=None):
        obs = check(obs).to(**self.tpdv)
        cent_obs = check(cent_obs).to(**self.tpdv)
        inps = torch.cat([obs, cent_obs], dim=-1)
        return self.act.get_dists(self.base(inps), available_actions)

    def evaluate_actions(self, obs, action, cent_obs, available_actions=None):
        action_dists = self.forward(obs, cent_obs, available_actions)
        action = check(action).to(**self.tpdv)
        return action_dists.log_prob(action.squeeze(-1)).unsqueeze(-1)


class HDDNetwork(object):
    def __init__(self, act_space, obs_space, cent_obs_space, args, device=torch.device("cpu")):
        # Used to esimate p(a_{t}^{i}|o_{t}^{i}, s_{t+1}^{j})
        self.hdd_epoch = args.hdd_epoch
        self.num_objects = args.num_objects
        self.num_mini_batch = args.num_mini_batch
        self.max_grad_norm = args.max_grad_norm
        self.hdd_lr = args.hdd_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_hdd_active_masks = args.use_hdd_active_masks

        self.hdd = HDD(act_space, obs_space, cent_obs_space, args, device=device)
        self.hdd_optim = torch.optim.Adam(self.hdd.parameters(),
                                          lr=self.hdd_lr,
                                          eps=self.opti_eps,
                                          weight_decay=self.weight_decay)

        self.tpdv = dict(dtype=torch.float32, device=device)

    def update(self, buffer):
        infos = {}
        infos["hdd_loss"] = []
        infos["hdd_grad_norm"] = []

        for object_i in range(self.num_objects):
            for _ in range(self.hdd_epoch):
                data_generator = buffer.hdd_generator(object_index=object_i, num_mini_batch=self.num_mini_batch)
                for sample in data_generator:
                    obs_batch, actions_batch, substates_batch, available_actions_batch, active_masks_batch = sample
                    active_masks_batch = check(active_masks_batch).to(**self.tpdv)

                    hdd_loss = - self.hdd.evaluate_actions(obs_batch, actions_batch, substates_batch, available_actions_batch)

                    if self._use_hdd_active_masks:
                        hdd_loss = (hdd_loss * active_masks_batch).sum() / active_masks_batch.sum()
                    else:
                        hdd_loss = hdd_loss.mean()

                    self.hdd_optim.zero_grad()
                    hdd_loss.backward()
                    if self._use_max_grad_norm:
                        hdd_grad_norm = nn.utils.clip_grad_norm_(self.hdd.parameters(), self.max_grad_norm)
                    else:
                        hdd_grad_norm = get_grad_norm(self.hdd.parameters())
                    self.hdd_optim.step()

                    infos["hdd_loss"].append(hdd_loss.item())
                    infos["hdd_grad_norm"].append(hdd_grad_norm.item())

        for k in infos.keys():
            infos[k] = np.mean(infos[k])

        return infos
    
    def evaluate_actions(self, obs, actions, object_substates, available_actions=None):
        log_probs = self.hdd.evaluate_actions(obs, actions, object_substates, available_actions)
        return torch.exp(log_probs).detach().cpu().numpy()
    
    def save(self, save_dir):
        torch.save(self.hdd.state_dict(), str(save_dir) + "/hdd_net.pt")
            
    def restore(self, model_dir):
        hdd_state_dict = torch.load(str(model_dir) + "/hdd_net.pt")
        self.hdd.load_state_dict(hdd_state_dict)