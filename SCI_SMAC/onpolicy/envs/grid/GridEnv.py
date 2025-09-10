"""Implements a model factory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from onpolicy.envs.grid.rooms import Rooms
from onpolicy.envs.grid.no_door_rooms import NoDoorRooms
from onpolicy.envs.grid.secret_rooms import SecretRooms
from onpolicy.envs.grid.push_box import PushBox


ENV_MAP = {
    'pass': Rooms(H=300, grid_size=30, n_actions=4, n_agents=2),
    'no_door_pass': NoDoorRooms(H=300, grid_size=30, n_actions=4, n_agents=2),
    'room30_ckpt': Rooms(H=300, grid_size=30, n_actions=4, n_agents=2, checkpoint=True),
    'secret_room': SecretRooms(H=300, grid_size=25, n_actions=4, n_agents=2),
    'secret_room_ckpt': SecretRooms(H=300, grid_size=25, n_actions=4, n_agents=2, checkpoint=True),
    'push_box': PushBox(H=300, grid_size=15, n_actions=4, n_agents=2),
    'push_box_ckpt': PushBox(H=300, grid_size=15, n_actions=4, n_agents=2, checkpoint=True),
}


def get_env(name):
  assert name in ENV_MAP
  return ENV_MAP[name]


def GridEnv(scenario_name):
    # Temporally supports pass env.
    assert scenario_name in ENV_MAP and scenario_name in ['pass', 'no_door_pass']
    env = ENV_MAP[scenario_name]
    return env
