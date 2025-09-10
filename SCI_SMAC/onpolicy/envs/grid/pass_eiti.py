import gym
import numpy as np
import copy


class Pass(gym.Env):
    def __init__(self, horizon=300, grid_size=30, n_actions=4, n_agents=2):
        self.size = grid_size       # 30
        self.map = np.zeros([self.size, self.size])     # 30x30

        self.map[:, self.size // 2] = -1        # The middle of the whole map
        self.map[int(self.size * 0.8), int(self.size * 0.2)] = 1        # switch one
        self.map[int(self.size * 0.2), int(self.size * 0.8)] = 1        # switch two

        self.landmarks = [np.array([int(self.size * 0.8), int(self.size * 0.2)]),
                          np.array([int(self.size * 0.2), int(self.size * 0.8)])]       # (24, 6) and (6, 24)

        self.door_open_interval = 8
        self.door_open = False
        self.door_open_step_count = 0

        self.n_agent = 2
        self.n_action = 4
        self.n_dim = 2

        self.state_n = [np.array([0, 0]) for _ in range(self.n_agent)]

        self.eye = np.eye(self.size)
        self.flag = np.eye(2)

        # Used by OpenAI baselines
        self.action_space = gym.spaces.Discrete(self.n_action)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=[args.size * 4])      # agent 0's x, y + agent 1's x y
        self.reward_range = (-100., 2000.)
        self.spec = 2

        self.t_step = 0

    def step(self, action_n, obs_a=False, obs_b=False, obs_c=False, obs_d=False):

        self.t_step += 1
        for i, action in enumerate(action_n):
            new_row = -1
            new_column = -1

            if action == 0:     # up
                new_row = max(self.state_n[i][0] - 1, 0)
                new_column = self.state_n[i][1]
            elif action == 1:       # right
                new_row = self.state_n[i][0]
                new_column = min(self.state_n[i][1] + 1, self.size - 1)
            elif action == 2:       # down
                new_row = min(self.state_n[i][0] + 1, self.size - 1)
                new_column = self.state_n[i][1]
            elif action == 3:       # left
                new_row = self.state_n[i][0]
                new_column = max(self.state_n[i][1] - 1, 0)

            if self.map[new_row][new_column] != -1:     # If not -1, then move to the target position
                self.state_n[i] = np.array([new_row, new_column])

        if self.door_open:
            if self.door_open_step_count >= self.door_open_interval:        # 当any agent触发switch打开door后，隔door_open_interval步数后door关闭
                # print('>>>>>> Door Closed')
                self.door_open = False
                self.map[int(self.size * 0.45):int(self.size * 0.55), self.size // 2] = -1
                self.door_open_step_count = 0
            else:
                self.door_open_step_count += 1

        if not self.door_open:
            for landmark_id, landmark in enumerate(self.landmarks):
                for i, state in enumerate(self.state_n):        # 只要有一个agent的state和door的coordinate重合，那么door开启
                    if (landmark == state).all():
                        if self.args.simple_env and landmark_id != i:
                            continue
                        # print('>>>>>>', i, 'Open the door.')
                        self.door_open = True
                        self.map[int(self.size * 0.45):int(self.size * 0.55), self.size // 2] = 0
                        self.door_open_step_count = 0
                        break

        # if obs_d:
        #     return self.observations_d()

        info = {'door': self.door_open, 'state': copy.deepcopy(self.state_n)}

        return self.obs_n(), self.reward(), self.done(), info

    def fix_reset(self):
        self.t_step = 0
        self.door_open = False
        self.door_open_step_count = 0

        self.state_n = [np.array([0, 0]) for _ in range(self.n_agent)]

        self.map[int(self.size * 0.45):int(self.size * 0.55), self.size // 2] = -1

        return self.obs_n()

    def reset(self, obs_d=False):
        self.t_step = 0
        self.door_open = False
        self.door_open_step_count = 0

        if (self.args.fix_start):
            self.state_n = [np.array([0, 0]) for _ in range(self.n_agent)]
        else:
            for i in range(self.n_agent):       # 两个agent位置随机初始化
                self.state_n[i][1] = np.random.randint(self.size // 2)
                self.state_n[i][0] = np.random.randint(self.size)

        self.map[int(self.size * 0.45):int(self.size * 0.55), self.size // 2] = -1      # 这里应该是指door

        # if obs_d:
        #     return self.observations_d()

        return self.obs_n()

    def random_reset(self, obs_d=False):
        self.t_step = 0
        self.door_open = False
        self.door_open_step_count = 0

        for i in range(self.n_agent):
            self.state_n[i][1] = np.random.randint(self.size // 2)
            self.state_n[i][0] = np.random.randint(self.size)

        self.map[int(self.size * 0.45):int(self.size * 0.55), self.size // 2] = -1

        # if obs_d:
        #     return self.observations_d()

        return self.obs_n()

    def local_state(self, i):
        return self.state_n[i]

    def local_states(self):
        return self.state_n

    def observation_a(self, i):
        return np.concatenate([self.state_n[i], np.array([int(self.door_open)])])

    def observations_a(self):
        return [self.observation_a(i) for i in range(self.n_agent)]

    def observation_b(self, i):
        return np.concatenate(self.state_n + [np.array([int(self.door_open)])])

    def observations_b(self):
        return [self.observation_b(i) for i in range(self.n_agent)]

    def observation(self, i):
        same_room = 0
        if self.state_n[i][1] < self.size // 2 and self.state_n[1 - i][1] < self.size // 2:
            same_room = 1

        if self.state_n[i][1] >= self.size // 2 and self.state_n[1 - i][1] >= self.size // 2:
            same_room = 1

        # indicator = same_room * 2 + self.door_open

        return np.concatenate([self.state_n[i], np.array([int(same_room), int(self.door_open)])])

    def obs_c(self, i):
        same_room = 0
        if self.state_n[i][1] < self.size // 2 and self.state_n[1 - i][1] < self.size // 2:
            same_room = 1

        if self.state_n[i][1] >= self.size // 2 and self.state_n[1 - i][1] >= self.size // 2:
            same_room = 1

        # indicator = same_room * 2 + self.door_open

        return np.concatenate([self.state_n[i], np.array([int(same_room), int(self.door_open)])])

    def obs_n(self):
        return [self.obs(i) for i in range(self.n_agent)]       # Two agents have the same observation

    # def observations_c(self):
    #     return [self.observation_c(i) for i in range(self.n_agent)]

    def obs(self, i):
        same_room = 0
        if self.state_n[i][1] < self.size // 2 and self.state_n[1 - i][1] < self.size // 2:     # 按照y坐标来判断两个智能体是否在同一room内
            same_room = 1

        if self.state_n[i][1] >= self.size // 2 and self.state_n[1 - i][1] >= self.size // 2:
            same_room = 1

        return np.concatenate([self.eye[self.state_n[0][0]], self.eye[self.state_n[0][1]],
                               self.eye[self.state_n[1][0]], self.eye[self.state_n[1][1]]]).copy()

    # self.flag[same_room],
    # self.flag[int(self.door_open)]]).copy()]

    # def observations_d(self):
    #     return [self.observation_d(i) for i in range(self.n_agent)]

    def reward(self):
        count = 0

        for i, state in enumerate(self.state_n):
            if state[1] > self.size // 2:
                count += 1      # 如果两个agent都进入右侧房间，那么count=2
            # print('>>>>>>', i, 'Pass.')

        return [(count >= 2) * 1000, (count >= 2) * 1000]

    def done(self):
        count = 0

        for state in self.state_n:
            if state[1] > self.size // 2:
                count += 1

        if count >= 2 or self.t_step >= self.args.episode_length:
            self.reset()
            return 1

        return 0

    def close(self):
        self.reset()