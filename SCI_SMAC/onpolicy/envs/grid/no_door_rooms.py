import gym
import numpy as np
import copy

TOP = 0
BOT = 1
LEFT = 2
RIGHT = 3


class Entity(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Door(Entity):
    def __init__(self, x, y):
        super(Door, self).__init__(x, y)
        self.open = 0


class NoDoorRooms(gym.Env):
    def __init__(self, H=40, grid_size=10, n_actions=4, n_agents=2, checkpoint=False):
        local_obs_space = gym.spaces.Box(low=-np.inf, high=+np.inf, shape=(3,), dtype=np.float32)  # [agent.x, agent.y. door.open]
        self.observation_space = [local_obs_space for _ in range(n_agents)]

        # Concat all agents' local obs as the global state
        share_obs_space = gym.spaces.Box(low=-np.inf, high=+np.inf, shape=(6,), dtype=np.float32)   # [x1, y1, door.open, x2, y2, door.open]
        self.share_observation_space = [share_obs_space for _ in range(n_agents)]

        local_act_space = gym.spaces.Discrete(n_actions)
        self.action_space = [local_act_space for _ in range(n_agents)]

        self.door_open_interval = 300
        self.door_open_step_count = 0

        self.init_agents = [Entity(1 + grid_size // 10, 1 + grid_size // 10), Entity(grid_size // 10, grid_size // 10)]     # Two agents.positions: (4, 4) and (3, 3)
        self.init_door = Door(grid_size // 2, grid_size // 2)       # door.position=(15, 15)
        self.switches = [Entity(grid_size // 10, int(grid_size * 0.8)), Entity(int(grid_size * 0.8), grid_size // 10)]      # Two switches.positions: (3, 24) and (24, 3)
        self.n_agents = n_agents
        self.grid_size = grid_size
        self.init_wall_map = np.zeros((grid_size, grid_size))
        self.init_wall_map[:, self.grid_size // 2] = 1          # 30x30 Grid [:, 15]=1, meaning that agents can not pass.
        self.H = H
        self.step_count = 0
        self.done = False
        self.agents, self.wall_map, self.door = None, None, None
        self.checkpoint = checkpoint
        self.n_ckpt_bins = 10
        # Below designs provide dense reward settings.
        self.ckpt_bins = {'left_switch': np.linspace(1.5 * (self.grid_size // 10) + 1, self._dist(self.init_agents[0], self.switches[0]) - 1, self.n_ckpt_bins),
                          'door': np.linspace(4, self._dist(self.init_agents[0], self.init_door) - 1, self.n_ckpt_bins),
                          'right_switch': np.linspace(1.5 * (self.grid_size // 10) + 1, self._dist(self.init_door, self.switches[1]) - 1, self.n_ckpt_bins)}
        self.ckpts = {'left_switch': [[False, bin] for bin in self.ckpt_bins['left_switch']],
                      'door': [[False, bin] for bin in self.ckpt_bins['door']],
                      'right_switch': [[False, bin] for bin in self.ckpt_bins['right_switch']]}
        self.success_rew = 3 if self.checkpoint else 100

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def reset(self):
        self.agents = copy.deepcopy(self.init_agents)
        self.wall_map = copy.deepcopy(self.init_wall_map)
        self.door = copy.deepcopy(self.init_door)

        door_radius = self.grid_size // 10  # 3
        self.door.open = 1
        self.wall_map[self.door.y, self.door.x] = 0  # Note that wall map is formalized as [y, x]
        self.wall_map[self.door.y - door_radius: self.door.y + door_radius + 1, self.door.x] = 0  # Why this operation?

        self.step_count = 0
        self.door_open_step_count = 0
        self.done = False
        self.ckpts = {'left_switch': [[False, bin] for bin in self.ckpt_bins['left_switch']],
                      'door': [[False, bin] for bin in self.ckpt_bins['door']],
                      'right_switch': [[False, bin] for bin in self.ckpt_bins['right_switch']]}
        # Return the concatenation of agent 0's pos, agent 1's pos and a binary variable indicating the status of door
        # return np.array([self.agents[0].x, self.agents[0].y, self.agents[1].x, self.agents[1].y, self.door.open])
        return [np.array([self.agents[0].x, self.agents[0].y, self.door.open]), np.array([self.agents[1].x, self.agents[1].y, self.door.open])]

    # def step(self, action):
    #     assert not self.done, "error: Trying to call step() after an episode is done"
    #     obs = []
    #     for agent_id, agent in enumerate(self.agents):
    #         self._update_agent_location(agent_id, action[agent_id])
    #         obs.extend([agent.x, agent.y])
    #     self._update_door_status()
    #     obs.append(self.door.open)
    #     self.step_count += 1
    #     rew = self._reward()
    #     self.done = True if self.step_count == self.H or rew >= self.success_rew else False
    #
    #     return np.array(obs), rew, self.done

    def step(self, action):
        assert not self.done, "error: Trying to call step() after an episode is done"
        for agent_id, agent in enumerate(self.agents):
            self._update_agent_location(agent_id, action[agent_id])

        # if self.door.open == 1:
        #     if self.door_open_step_count >= self.door_open_interval:        # 当any agent触发switch打开door后，隔door_open_interval步数后door关闭
        #         # print('>>>>>> Door Closed')
        #         self.door.open = 0
        #         # Close the door again.
        #         door_radius = self.grid_size // 10  # 3
        #         self.wall_map[self.door.x, self.door.y] = 1
        #         self.wall_map[self.door.y - door_radius: self.door.y + door_radius + 1, self.door.x] = 1
        #
        #         self.door_open_step_count = 0
        #     else:
        #         self.door_open_step_count += 1
        #
        # if self.door.open == 0:
        #     self._update_door_status()

        obs_n = [np.array([self.agents[0].x, self.agents[0].y, self.door.open]), np.array([self.agents[1].x, self.agents[1].y, self.door.open])]
        self.step_count += 1

        rew = self._reward()
        reward_n = [[rew]] * 2

        self.done = True if self.step_count == self.H or rew >= self.success_rew else False
        done_n = [self.done for _ in range(2)]

        info_n = [{} for _ in range(2)]

        return obs_n, reward_n, done_n, info_n

    # Action: 0-up, 1-bottom, 2-left, 3-right
    def _update_agent_location(self, agent_id, action):
        x, y = self.agents[agent_id].x, self.agents[agent_id].y
        if action == TOP and y > 0 and self.wall_map[y - 1, x] == 0:
            self.agents[agent_id].y -= 1
        elif action == BOT and y < self.grid_size - 1 and self.wall_map[y + 1, x] == 0:
            self.agents[agent_id].y += 1
        elif action == LEFT and x > 0 and self.wall_map[y, x - 1] == 0:
            self.agents[agent_id].x -= 1
        elif action == RIGHT and x < self.grid_size - 1 and self.wall_map[y, x + 1] == 0:
            self.agents[agent_id].x += 1

    def _update_door_status(self):
        door_radius = self.grid_size // 10      # 3
        for switch in self.switches:
            for agent in self.agents:
                if self._dist(agent, switch) <= 1.5 * door_radius:      # If distance(agent, switch)<=threshold, switch is activated and the door opens.
                    self.door.open = 1
                    self.wall_map[self.door.y, self.door.x] = 0     # Note that wall map is formalized as [y, x]
                    self.wall_map[self.door.y - door_radius: self.door.y + door_radius + 1, self.door.x] = 0    # Why this operation?

                    self.door_open_step_count = 0
                    return
        # self.door.open = 0
        # # Close the door again.
        # self.wall_map[self.door.x, self.door.y] = 1
        # self.wall_map[self.door.y - door_radius: self.door.y + door_radius + 1, self.door.x] = 1

    def _dist(self, e1, e2):
        return np.sqrt((e1.x - e2.x) ** 2 + (e1.y - e2.y) ** 2)

    def _reward(self):
        rew = 0
        if self.checkpoint:
            rew += self._checkpoint_rew()
        for agent in self.agents:
            if agent.x < (self.grid_size / 2 + 1):      # If any agent still lies in the left room, a team reward of 0 is emitted.
                return rew
        rew += self.success_rew
        return rew

    def _checkpoint_rew(self):
        rew = 0

        # door
        if self.door.open:
            for ckpt in self.ckpts['door']:
                for agent in self.agents:
                    if ckpt[0]:
                        continue
                    if agent.x < (self.grid_size / 2 + 1):
                        if self._dist(agent, self.door) <= ckpt[1]:
                            rew += 0.1
                            ckpt[0] = True

        # Right switch
        for ckpt in self.ckpts['right_switch']:
            for agent in self.agents:
                if ckpt[0]:
                    continue
                if agent.x >= (self.grid_size / 2 + 1):
                    if self._dist(agent, self.switches[1]) <= ckpt[1]:
                        rew += 0.1
                        ckpt[0] = True

        return rew