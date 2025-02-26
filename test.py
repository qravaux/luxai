from src.luxai_s3.wrappers import LuxAIS3GymEnv
from policies import Luxai_Agent
import torch
import numpy as np
from src.luxai_s3.pygame_render import LuxAIPygameRenderer

renderer = LuxAIPygameRenderer()

env = LuxAIS3GymEnv(numpy_output=True)
obs,_ = env.reset(seed=26)

renderer.render(env.state,env.env_params)

ep_params = {}
ep_params['unit_move_cost'] = env.env_params.unit_move_cost
ep_params['unit_sap_cost'] = env.env_params.unit_sap_cost
ep_params['unit_sap_range'] = env.env_params.unit_sap_range
ep_params['unit_sensor_range'] = env.env_params.unit_sensor_range
print(ep_params)

policy_0 = Luxai_Agent('player_0')
policy_1 = Luxai_Agent('player_1')

#policy_0.load_state_dict(torch.load("policy/exploit_1_unit_A2C/policy_0_epoch_900.pth", weights_only=True))
#policy_1.load_state_dict(torch.load("policy/exploit_1_unit_A2C/policy_1_epoch_900.pth", weights_only=True))


state_maps_0, state_features_0 = policy_0.obs_to_state(obs['player_0'],ep_params)
state_maps_1, state_features_1 = policy_1.obs_to_state(obs['player_1'],ep_params)

for step in range(505) :

    if step == 0:
        points_0 = obs['player_0']['team_points'][0]
        points_1 = obs['player_1']['team_points'][1]
        previous_obs = obs

    else :
        points_0 = (obs['player_0']['team_points'][0] - previous_obs['player_0']['team_points'][0])
        points_1 = (obs['player_1']['team_points'][1] - previous_obs['player_1']['team_points'][1])
        previous_obs = obs

    show = step==480

    if show :
        print(f"player 0 : {obs['player_0']['team_points'][0]} player 1 : {obs['player_0']['team_points'][1]}")
        renderer.render(env.state,env.env_params)
    
    state_maps_0 ,state_features_0 = policy_0.obs_to_state(obs['player_0'],ep_params,points_0,state_maps_0,show=show)
    action_0,_,_,_,_,_ = policy_0(state_maps_0 ,state_features_0,obs['player_0'],ep_params)

    state_maps_1 ,state_features_1 = policy_1.obs_to_state(obs['player_1'],ep_params,points_1,state_maps_1,show=show)
    action_1,_,_,_,_,_ = policy_1(state_maps_1 ,state_features_1,obs['player_1'],ep_params)

    action = dict(player_0=np.array(action_0, dtype=int), player_1=np.array(action_1, dtype=int))
    obs, reward, truncated, done, info = env.step(action)

   