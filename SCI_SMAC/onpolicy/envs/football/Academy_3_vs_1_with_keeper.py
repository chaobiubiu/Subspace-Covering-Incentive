import random

import gfootball.env as football_env
from gym import spaces
import numpy as np


class Academy_3_vs_1_with_Keeper(object):

    def __init__(self, args):
        assert args.num_agents == 3, "Academy_3_vs_1_with_Keeper must follow 3-agents setting."
        self.num_agents = args.num_agents
        self.n_agents = args.num_agents
        self.scenario_name = args.scenario_name

        # make env
        if not (args.use_render and args.save_videos):
            self.env = football_env.create_environment(
                env_name=args.scenario_name,
                stacked=args.use_stacked_frames,
                representation=args.representation,
                rewards=args.rewards,
                number_of_left_players_agent_controls=args.num_agents,
                number_of_right_players_agent_controls=0,
                channel_dimensions=(args.smm_width, args.smm_height),
                render=(args.use_render and args.save_gifs)
            )
        else:
            # render env and save videos
            self.env = football_env.create_environment(
                env_name=args.scenario_name,
                stacked=args.use_stacked_frames,
                representation=args.representation,
                rewards=args.rewards,
                number_of_left_players_agent_controls=args.num_agents,
                number_of_right_players_agent_controls=0,
                channel_dimensions=(args.smm_width, args.smm_height),
                # video related params
                write_full_episode_dumps=True,
                render=True,
                write_video=True,
                dump_frequency=1,
                logdir=args.video_dir
            )

        self.obs_dim = 26

        assert args.episode_length == 150, "Football env must run under the setting of 150 episode limit."
        self.episode_limit = args.episode_limit
        self.time_step = 0

        self.max_steps = self.env.unwrapped.observation()[0]["steps_left"]
        self.remove_redundancy = args.remove_redundancy
        self.zero_feature = args.zero_feature
        self.share_reward = args.share_reward
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        obs_space_low = self.env.observation_space.low[0][:self.obs_dim]
        obs_space_high = self.env.observation_space.high[0][:self.obs_dim]

        for idx in range(self.num_agents):
            self.action_space.append(spaces.Discrete(
                n=self.env.action_space[idx].n
            ))
            self.observation_space.append(spaces.Box(
                low=obs_space_low,
                high=obs_space_high,
                dtype=self.env.observation_space.dtype
            ))
            self.share_observation_space.append(spaces.Box(
                low=obs_space_low,
                high=obs_space_high,
                dtype=self.env.observation_space.dtype
            ))

        self.n_actions = self.action_space[0].n

        # For the player, its local info contains (x,y) position and (x,y) direction.
        # For the ball, its local info contains (x,y,z) position and (x,y,z) direction.
        self.nf_al, self.nf_en = 4, 6

    def get_simple_obs(self, index=-1):
        full_obs = self.env.unwrapped.observation()[0]
        simple_obs = []

        if index == -1:
            # global state, absolute position
            simple_obs.append(full_obs['left_team'][-self.n_agents:].reshape(-1))       # 2x3
            simple_obs.append(full_obs['left_team_direction'][-self.n_agents:].reshape(-1))     # 2x3

            simple_obs.append(full_obs['right_team'].reshape(-1))       # 2x2
            simple_obs.append(full_obs['right_team_direction'].reshape(-1))     # 2x2

            simple_obs.append(full_obs['ball'])     # 3
            simple_obs.append(full_obs['ball_direction'])       # 3

        else:
            # local state, relative position
            ego_position = full_obs['left_team'][-self.n_agents + index].reshape(-1)
            simple_obs.append(ego_position)
            simple_obs.append((np.delete(full_obs['left_team'][-self.n_agents:], index, axis=0) - ego_position).reshape(-1))     # The relative position of team players except for agent i

            simple_obs.append(full_obs['left_team_direction'][-self.n_agents + index].reshape(-1))
            simple_obs.append(np.delete(full_obs['left_team_direction'][-self.n_agents:], index, axis=0).reshape(-1))       # The directions of team players including both current agent and other teams.

            simple_obs.append((full_obs['right_team'] - ego_position).reshape(-1))
            simple_obs.append(full_obs['right_team_direction'].reshape(-1))     # Right team relative position and their directions.

            simple_obs.append(full_obs['ball'][:2] - ego_position)      # Relative x, y of the ball relative to current agent.
            simple_obs.append(full_obs['ball'][-1].reshape(-1))         # z of the ball.
            simple_obs.append(full_obs['ball_direction'])               # Ball direction.

        simple_obs = np.concatenate(simple_obs)
        return simple_obs

    def reset(self):
        """Returns initial observations and states."""
        self.env.reset()
        self.time_step = 0
        obs = np.array([self.get_simple_obs(i) for i in range(self.num_agents)])
        states = np.array([self.get_global_state() for i in range(self.num_agents)])

        return obs, states, self.get_avail_actions()

    def get_global_state(self):
        return self.get_simple_obs(-1)

    def check_if_done(self):
        cur_obs = self.env.unwrapped.observation()[0]
        ball_loc = cur_obs['ball']
        ours_loc = cur_obs['left_team'][-self.n_agents:]

        if ball_loc[0] < 0 or any(ours_loc[:, 0] < 0):
            return True

        return False

    def step(self, actions):
        """Returns reward, terminated, info."""
        self.time_step += 1
        _, original_rewards, done, info = self.env.step(actions)
        rewards = list(original_rewards)

        if self.time_step >= self.episode_limit:
            done = True

        if self.check_if_done():
            done = True

        obs = np.array([self.get_simple_obs(i) for i in range(self.num_agents)])
        state = np.array([self.get_global_state() for i in range(self.num_agents)])
        env_done = np.array([done] * self.num_agents)
        info = self._info_wrapper(info)

        if sum(rewards) <= 0:
            return obs, state, [[-int(done)]] * self.num_agents, env_done, info, self.get_avail_actions()

        return obs, state, [[100]] * self.num_agents, env_done, info, self.get_avail_actions()

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        return [[1 for _ in range(self.n_actions)] for agent_id in range(self.num_agents)]

    def seed(self, seed=None):
        if seed is None:
            random.seed(1)
        else:
            random.seed(seed)

    def close(self):
        self.env.close()

    def _info_wrapper(self, info):
        state = self.env.unwrapped.observation()
        info.update(state[0])
        info["max_steps"] = self.max_steps
        info["active"] = np.array([state[i]["active"] for i in range(self.num_agents)])
        info["designated"] = np.array([state[i]["designated"] for i in range(self.num_agents)])
        info["sticky_actions"] = np.stack([state[i]["sticky_actions"] for i in range(self.num_agents)])
        return info