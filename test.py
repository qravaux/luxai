from src.luxai_s3.wrappers import LuxAIS3GymEnv
from policies import Luxai_Agent
import torch
import numpy as np
from src.luxai_s3.pygame_render import LuxAIPygameRenderer

renderer = LuxAIPygameRenderer()

env = LuxAIS3GymEnv(numpy_output=True)
obs,_ = env.reset(seed=20)

renderer.render(env.state,env.env_params)

ep_params = {}
ep_params['unit_move_cost'] = env.env_params.unit_move_cost
ep_params['unit_sap_cost'] = env.env_params.unit_sap_cost
ep_params['unit_sap_range'] = env.env_params.unit_sap_range
ep_params['unit_sensor_range'] = env.env_params.unit_sensor_range
print(ep_params)

policy_0 = Luxai_Agent('player_0')
policy_1 = Luxai_Agent('player_1')

policy_0.load_state_dict(torch.load("policy/A2C_unit_reward/policy_0_epoch_30000.pth", weights_only=True))
policy_1.load_state_dict(torch.load("policy/A2C_unit_reward/policy_1_epoch_30000.pth", weights_only=True))


state_maps_0, state_features_0 = policy_0.obs_to_state(obs['player_0'],ep_params)
state_maps_1, state_features_1 = policy_1.obs_to_state(obs['player_1'],ep_params)

for step in range(505) :
    show = (step%2)==0

    if show : 

        renderer.render(env.state,env.env_params)

    if step == 0:
        points_0 = obs['player_0']['team_points'][0]
        points_1 = obs['player_1']['team_points'][1]
        previous_obs = obs

    else :
        points_0 = (obs['player_0']['team_points'][0] - previous_obs['player_0']['team_points'][0])
        points_1 = (obs['player_1']['team_points'][1] - previous_obs['player_1']['team_points'][1])
        previous_obs = obs
    print(points_0,points_1)
    #print(obs['player_0']['relic_nodes_mask'])
    state_maps_0 ,state_features_0 = policy_0.obs_to_state(obs['player_0'],ep_params,points_0,state_maps_0,show=False)
    action_0, value, mask_action, mask_dx, mask_dy, log_prob = policy_0(state_maps_0 ,state_features_0,obs['player_0'],ep_params)

    state_maps_1 ,state_features_1 = policy_1.obs_to_state(obs['player_1'],ep_params,points_1,state_maps_1,show=False)
    action_1, value, mask_action, mask_dx, mask_dy, log_prob = policy_1(state_maps_1 ,state_features_1,obs['player_1'],ep_params)

    action = dict(player_0=np.array(action_0, dtype=int), player_1=np.array(action_1, dtype=int))
    obs, reward, truncated, done, info = env.step(action)

    


   