import wandb
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from onpolicy.utils.util import get_shape_from_obs_space
from onpolicy.utils.shared_buffer import SharedReplayBuffer, HindsightReplayBuffer, HindsightBuffer


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        self.num_objects = self.all_args.num_objects
        self.num_entities = self.all_args.num_entities

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

        assert self.use_centralized_V, "we recommend to use ppo with centralized critic."

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        print("obs_space: ", self.envs.observation_space)
        print("share_obs_space: ", self.envs.share_observation_space)
        print("act_space: ", self.envs.action_space)

        cent_obs_shape = get_shape_from_obs_space(share_observation_space)
        if len(cent_obs_shape) == 3:
            raise NotImplementedError
        else:
            cent_obs_dim = cent_obs_shape[0]

        if self.all_args.env_name == "SparseStarCraft2":
            assert not self.all_args.use_state_agent, "we do not use the customized state for the SMAC benchmark."
            self.nf_al, self.nf_en = self.envs.nf_al, self.envs.nf_en
            self.al_en_features = self.nf_al * self.num_agents + self.nf_en * (self.num_entities - self.num_agents)
            self.other_features = cent_obs_dim - self.al_en_features
            self.cent_obs_dim = cent_obs_dim
        elif self.all_args.env_name == "Football":
            self.nf_al, self.nf_en = self.envs.nf_al, self.envs.nf_en       # Respectively denote feature_dim of the player and the ball in the GRF task.
            self.cent_obs_dim = cent_obs_dim
        else:
            raise NotImplementedError

        # policy network
        self.policy = Policy(self.all_args, self.envs.observation_space[0], share_observation_space, self.envs.action_space[0], device=self.device)

        if self.model_dir is not None:
            self.restore(self.model_dir)

        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, device=self.device)
        
        # buffer
        self.buffer = HindsightReplayBuffer(self.all_args,
                                            self.num_agents,
                                            self.envs.observation_space[0],
                                            share_observation_space,
                                            self.envs.action_space[0])

    def get_object_masks(self):
        entity_masks = self.policy.get_entity_masks()       # (num_objects, num_entities), contains 0 or 1
        object_masks = [np.zeros(self.cent_obs_dim, dtype=np.float32) for _ in range(self.num_objects)]
        if self.all_args.env_name == "SparseStarCraft2":
            # For SMAC, the state contains [ally1.state, ally2.state, ally3.state, ..., enemy1.state, enemy2.state, enemy3.state, ...]
            for i in range(self.num_objects):
                for j in range(self.num_entities):
                    if j < self.num_agents:
                        object_masks[i][self.nf_al*j: self.nf_al*(j+1)] = entity_masks[i][j]
                    else:
                        # Before 2024-5-12, there exists Error.
                        # object_masks[i][self.nf_en*j: self.nf_en*(j+1)] = entity_masks[i][j]
                        object_masks[i][(self.nf_al*self.num_agents+self.nf_en*(j-self.num_agents)): (self.nf_al*self.num_agents+self.nf_en*(j-self.num_agents+1))] = entity_masks[i][j]
        # elif self.all_args.env_name == "Football":
        #     # For GRF, the state contains [(x,y) position of n-agents left players, (x,y) direction of n_agents left players,
        #     # (x,y) position of right players, (x,y) direction of right players, (x,y,z) position and (x,y,z) direction of the ball]
        #     for i in range(self.num_objects):
        #         for j in range(self.num_entities):
        #             if j < self.num_agents:     # Number of left players
        #                 object_masks[i][2*j: 2*(j+1)] = entity_masks[i][j]
        #                 object_masks[i][(2*self.num_agents + 2*j): (2*self.num_agents + 2*(j+1))] = entity_masks[i][j]
        #             elif self.num_agents <= j < self.num_entities - 1:
        #                 object_masks[i][(4*self.num_agents + 2*(j-self.num_agents)): (4*self.num_agents + 2*(j-self.num_agents+1))] = entity_masks[i][j]
        #                 object_masks[i][(4*self.num_agents+2*(self.num_entities-1-self.num_agents)+2*(j-self.num_agents)): (4*self.num_agents+2*(self.num_entities-1-self.num_agents)+2*(j-self.num_agents+1))] = entity_masks[i][j]
        #             else:
        #                 object_masks[i][4*self.num_agents+4*(self.num_entities-1-self.num_agents):] = entity_masks[i][j]
        elif self.all_args.env_name == "Football":
            # For GRF, the state contains [(x,y) position of 11 left players, (x,y) direction of 11 left players,
            # (x,y) position of 11 right players, (x,y) direction of 11 right players, (x,y,z) position and (x,y,z) direction of the ball]
            # one-hot encoding of ball ownship (none, left, right), one-hot encoding of which player is active, one-hot encoding of game mode
            # 22 + 22 + 22 + 22 + 6 + 3 + 11 + 7 = 115
            # Note: we only consider info about left team players, right team players and the ball.
            for i in range(self.num_objects):
                for j in range(self.num_entities):
                    if j < 11:     # Number of left players
                        object_masks[i][2*j: 2*(j+1)] = entity_masks[i][j]
                        object_masks[i][(2*11 + 2*j): (2*11 + 2*(j+1))] = entity_masks[i][j]
                    elif 11 <= j < self.num_entities - 1:
                        object_masks[i][(4*11 + 2*(j-11)): (4*11 + 2*(j-11+1))] = entity_masks[i][j]
                        object_masks[i][(4*11+2*11+2*(j-11)): (4*11+2*11+2*(j-11+1))] = entity_masks[i][j]
                    else:
                        object_masks[i][4*11+4*11: (4*11+4*11+6)] = entity_masks[i][j]
        else:
            raise Exception("Unknown env name for object mask generation.")
        return object_masks

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError

    def compute(self):
        # State novelties calculated by self.policy.state_rnd
        state_novelties = self.policy.get_state_rnd_results(self.buffer.share_obs)      # shape=(episode_length+1, n_rollout_threads, num_agents, 1)
        self.buffer.state_novels = state_novelties

        # Sub-state novelties calculated by self.policy.sub_state_rnd
        substates = []
        substate_novelties = []
        for i in range(self.num_objects):
            # curr_substate = self.buffer.share_obs * self.object_masks[i]        # (episode_length+1, n_rollout_threads, num_agents, cent_obs_shape)
            curr_substate = self.buffer.share_obs * self.get_object_masks()[i]  # (episode_length+1, n_rollout_threads, num_agents, cent_obs_shape)
            curr_substate_novelty = self.policy.get_substate_rnd_results(curr_substate)     # shape=(episode_length+1, n_rollout_threads, num_agents, 1)
            substates.append(curr_substate)
            substate_novelties.append(curr_substate_novelty)
        substates = np.stack(substates, axis=0)     # (num_objects, episode_length+1, n_rollout_threads, num_agents, cent_obs_shape)
        substate_novelties = np.stack(substate_novelties, axis=0)       # (num_objects, episode_length+1, n_rollout_threads, num_agents, 1)

        self.buffer.sub_states = substates
        self.buffer.sub_state_novels = substate_novelties

        # Update policy.hdd_net p(a_{t}^{i}|o_{t}^{i}, s_{t+1}^{j}) for all possible objects j
        hdd_train_infos = self.trainer.hdd_update(self.buffer)

        # Estimate p(a_{t}^{i}|o_{t}^{i}, s_{t+1}^{j}) * novel(s_{t+1}^{j}) as hdd_advantages (i.e., weighted sub-state novelties)
        self.trainer.cal_hdd_advantages(self.buffer)

        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                     np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                     np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values,
                                    self.trainer.value_normalizer,
                                    state_novel_coef=self.all_args.state_novel_coef,
                                    ir_coef=self.all_args.ir_coef)

        return hdd_train_infos

    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer, object_masks=self.get_object_masks())
        self.buffer.after_update()
        return train_infos

    def save(self, episode=0):
        """Save policy's actor and critic networks."""
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")
        policy_mask_generator = self.trainer.policy.mask_generator
        torch.save(policy_mask_generator.state_dict(), str(self.save_dir) + "/mask_generator.pt")

    def restore(self, model_dir):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.all_args.use_render:
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt')
            self.policy.critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                # self.writter.add_scalars(k, {k: v}, total_num_steps)
                self.writter.add_scalar(k, v, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    # self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
                    self.writter.add_scalar(k, np.mean(v), total_num_steps)