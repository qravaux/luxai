from src.luxai_s3.wrappers import LuxAIS3GymEnv
import pygame
import time
from policies import Luxai_Agent
import torch
import numpy as np
from src.luxai_s3.pygame_render import LuxAIPygameRenderer

renderer = LuxAIPygameRenderer()

env = LuxAIS3GymEnv(numpy_output=True)
obs,_ = env.reset(seed=12)

renderer.render(env.state,env.env_params)

ep_params = {}
ep_params['unit_move_cost'] = env.env_params.unit_move_cost
ep_params['unit_sap_cost'] = env.env_params.unit_sap_cost
ep_params['unit_sap_range'] = env.env_params.unit_sap_range
ep_params['unit_sensor_range'] = env.env_params.unit_sensor_range

relic_node_positions = []
discovered_relic_nodes_ids = set()
unit_explore_locations = dict()

policy_0 = Luxai_Agent('player_0')
policy_1 = Luxai_Agent('player_1')

policy_0.load_state_dict(torch.load("policy/experiment_12/policy_0_epoch_360.pth", weights_only=True))
policy_1.load_state_dict(torch.load("policy/experiment_12/policy_1_epoch_360.pth", weights_only=True))

for step in range(505) :

    state_maps ,state_features = policy_0.obs_to_state(obs['player_0'],ep_params)
    action_0,_,_,_,_ = policy_0(state_maps ,state_features,obs['player_0'],ep_params)

    state_maps ,state_features = policy_0.obs_to_state(obs['player_1'],ep_params)
    action_1,_,_,_,_ = policy_1(state_maps ,state_features,obs['player_1'],ep_params)

    action = dict(player_0=np.array(action_0, dtype=int), player_1=np.array(action_1, dtype=int))
    obs, reward, truncated, done, info = env.step(action)

    if step % 10 == 0 :
        renderer.render(env.state,env.env_params)