import numpy as np
import torch
import torch.nn as nn
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check


class R_MAPPO():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.num_objects = args.num_objects
        self.use_weight_one = args.use_weight_one
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        
        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1).to(self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def cal_state_rnd_loss(self, cent_obs_batch, active_masks_batch):
        predict_features, target_features = self.policy.state_rnd(cent_obs_batch)
        state_rnd_loss = torch.square(predict_features - target_features).mean(dim=-1, keepdim=True)

        state_rnd_loss = (state_rnd_loss * active_masks_batch).sum() / active_masks_batch.sum()

        return state_rnd_loss

    def cal_substate_rnd_loss(self, substates_batch, active_masks_batch):
        predict_features, target_features = self.policy.sub_state_rnd(substates_batch)
        substate_rnd_loss = torch.square(predict_features - target_features).mean(dim=-1, keepdim=True)

        substate_rnd_loss = (substate_rnd_loss * active_masks_batch).sum() / active_masks_batch.sum()

        return substate_rnd_loss

    def ppo_update(self, sample, object_masks, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        # State rnd update
        state_rnd_loss = self.cal_state_rnd_loss(share_obs_batch, active_masks_batch)

        self.policy.state_rnd_optimizer.zero_grad()

        state_rnd_loss.backward()

        if self._use_max_grad_norm:
            state_rnd_grad_norm = nn.utils.clip_grad_norm_(self.policy.state_rnd.parameters(), self.max_grad_norm)
        else:
            state_rnd_grad_norm = get_gard_norm(self.policy.state_rnd.parameters())

        self.policy.state_rnd_optimizer.step()

        # Substate rnd update, we use object_masks to construct, instead of selecting from the buffer.substates for convenience.
        substate_rnd_loss = torch.tensor(0.0, device=self.device)
        for i in range(self.num_objects):
            curr_substates_batch = share_obs_batch * object_masks[i]
            curr_substate_rnd_loss = self.cal_substate_rnd_loss(curr_substates_batch, active_masks_batch)
            substate_rnd_loss += curr_substate_rnd_loss

        self.policy.sub_state_rnd_optimizer.zero_grad()

        substate_rnd_loss.backward()

        if self._use_max_grad_norm:
            substate_rnd_grad_norm = nn.utils.clip_grad_norm_(self.policy.sub_state_rnd.parameters(), self.max_grad_norm)
        else:
            substate_rnd_grad_norm = get_gard_norm(self.policy.sub_state_rnd.parameters())

        self.policy.sub_state_rnd_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, state_rnd_loss, \
               state_rnd_grad_norm, substate_rnd_loss, substate_rnd_grad_norm

    def hdd_update(self, buffer):
        return self.policy.hdd.update(buffer)

    def cal_state_hdd_ratio(self, buffer):
        substates = buffer.sub_states[:, 1:]        # shape=(num_objects, episode_length, n_rollout_threads, num_agents, cent_obs_shape)

        h_probs = []
        for i in range(self.num_objects):
            # p(a_{t}^{i}|o_{t}^{i}, s_{t+1}^{j})
            h_prob_i = self.policy.hdd.evaluate_actions(buffer.obs[:-1], buffer.actions, substates[i], buffer.available_actions[:-1])
            h_probs.append(h_prob_i)
        h_probs = np.stack(h_probs, axis=0)     # (num_objects, episode_length, n_rollout_threads, num_agents, 1)
        a_probs = np.exp(buffer.action_log_probs)   # (episode_length, n_rollout_threads, num_agents, 1)
        buffer.ratios[:] = a_probs / h_probs

    def cal_hdd_advantages(self, buffer):
        self.cal_state_hdd_ratio(buffer)

        weight = - np.log(buffer.ratios + 1e-6)      # log(h_probs / a_probs), (num_objects, episode_length, n_rollout_threads, num_agents, 1)

        if self.use_weight_one:
            weight = np.ones_like(weight)

        # Note here we use the immediate novelties of s_{t+1}^{j}
        novel_returns = buffer.sub_state_novels[:, 1:]      # (num_objects, episode_length, n_rollout_threads, num_agents, 1)
        hdd_advantages = weight * novel_returns

        hdd_advantages *= buffer.active_masks[:-1]

        buffer.hdd_advantages = hdd_advantages.sum(axis=0)  # Respectively calculate weighted sub-state novelties, then sum up.

    # def update_mask_generator(self):
    #     entity_masks = self.policy.get_entity_masks(is_train=True)  # (num_objects, num_entities)
    #     entity_masks_1 = entity_masks.unsqueeze(dim=1)  # (num_objects, 1, num_entities)
    #     entity_masks_2 = entity_masks.unsqueeze(dim=0).clone().detach()  # (1, num_objects, num_entities)
    #     # Maximize the difference between them
    #     mask_div_loss = - ((entity_masks_1 - entity_masks_2) ** 2).mean()  # (num_objects, num_objects, num_entities)-->mean
    #
    #     self.policy.mask_generator_optimizer.zero_grad()
    #     mask_div_loss.backward()
    #
    #     if self._use_max_grad_norm:
    #         mask_generator_grad_norm = nn.utils.clip_grad_norm_(self.policy.mask_generator.parameters(),
    #                                                             self.max_grad_norm)
    #     else:
    #         mask_generator_grad_norm = get_gard_norm(self.policy.mask_generator.parameters())
    #
    #     self.policy.mask_generator_optimizer.step()
    #
    #     return mask_div_loss, mask_generator_grad_norm

    def calculate_hamming_distance(self, entity_masks):
        batch_size = entity_masks.size(0)
        num_entities = entity_masks.size(1)
        expanded_masks_1 = entity_masks.unsqueeze(1).expand(batch_size, batch_size, num_entities)
        expanded_masks_2 = entity_masks.unsqueeze(0).expand(batch_size, batch_size, num_entities)

        hamming_distance_loss = - torch.abs(expanded_masks_1 - expanded_masks_2).sum(dim=-1).mean()
        return hamming_distance_loss

    def update_mask_generator(self):
        entity_masks = self.policy.get_entity_masks(is_train=True)  # (num_objects, num_entities)
        hamming_distance_loss = self.calculate_hamming_distance(entity_masks)

        self.policy.mask_generator_optimizer.zero_grad()
        hamming_distance_loss.backward()

        if self._use_max_grad_norm:
            mask_generator_grad_norm = nn.utils.clip_grad_norm_(self.policy.mask_generator.parameters(),
                                                                self.max_grad_norm)
        else:
            mask_generator_grad_norm = get_gard_norm(self.policy.mask_generator.parameters())

        self.policy.mask_generator_optimizer.step()

        return hamming_distance_loss, mask_generator_grad_norm

    def train(self, buffer, object_masks, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['state_rnd_loss'] = 0
        train_info['state_rnd_grad_norm'] = 0
        train_info['substate_rnd_loss'] = 0
        train_info['substate_rnd_grad_norm'] = 0

        train_info['entity_mask_loss'] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:

                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, state_rnd_loss, \
                state_rnd_grad_norm, substate_rnd_loss, substate_rnd_grad_norm = self.ppo_update(sample, object_masks, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()
                train_info['state_rnd_loss'] += state_rnd_loss.item()
                train_info['state_rnd_grad_norm'] += state_rnd_grad_norm.item()
                train_info['substate_rnd_loss'] += substate_rnd_loss.item()
                train_info['substate_rnd_grad_norm'] += substate_rnd_grad_norm.item()

            mask_div_loss, mask_generator_grad_norm = self.update_mask_generator()
            train_info['entity_mask_loss'] += mask_div_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            if k not in ['entity_mask_loss']:
                train_info[k] /= num_updates
            else:
                train_info[k] /= self.ppo_epoch

        train_info['ext_rew_mean'] = np.mean(buffer.rewards)
        train_info['state_novels_mean'] = np.mean(buffer.state_novels)
        train_info['substate_novels_mean'] = np.mean(buffer.sub_state_novels)
        train_info['hdd_advantages_mean'] = np.mean(buffer.hdd_advantages)
 
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()
        # self.policy.state_rnd.train()
        # self.policy.sub_state_rnd.train()
        # self.policy.hdd.hdd.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
        # self.policy.state_rnd.eval()
        # self.policy.sub_state_rnd.eval()
        # self.policy.hdd.hdd.eval()