from src.luxai_s3.env import LuxAIS3Env
from src.luxai_s3.params import EnvParams, env_params_ranges
from policies import Luxai_Agent
import torch
import numpy as np
import jax
from src.luxai_s3.pygame_render import LuxAIPygameRenderer

def random_params(rng_key) : 
    randomized_game_params = dict()
    for k, v in env_params_ranges.items():
        rng_key, subkey = jax.random.split(rng_key)
        randomized_game_params[k] = jax.random.choice(
            subkey, jax.numpy.array(v)
        ).item()
    params = EnvParams(**randomized_game_params)
    return params

renderer = LuxAIPygameRenderer()

env = LuxAIS3Env(auto_reset=False)
rng = jax.random.key(0)
rng, key_reset = jax.random.split(rng, 2)
params = random_params(key_reset)
obs,state = env.reset(key_reset,params)

renderer.render(state,params)

policy_0 = Luxai_Agent('player_0')
policy_1 = Luxai_Agent('player_1')

policy_0.load_state_dict(torch.load("policy/A2C_unit_reward/policy_0_epoch_30000.pth", weights_only=True))
policy_1.load_state_dict(torch.load("policy/A2C_unit_reward/policy_1_epoch_30000.pth", weights_only=True))


state_maps_0, state_features_0 = policy_0.obs_to_state(obs['player_0'],params)
state_maps_1, state_features_1 = policy_1.obs_to_state(obs['player_1'],params)

for step in range(505) :
    show = step%100== 0

    if step%10==0 :
        renderer.render(state,params)

    if step == 0:
        reward_memory = {'player_0':0,'player_1':0}
        points_0 = int(obs['player_0'].team_points[0])
        points_1 = int(obs['player_1'].team_points[1])
        previous_obs = obs

    elif reward['player_0'] > reward_memory['player_0'] :
        reward_memory = reward
        points_0 = int(obs['player_0'].team_points[0])
        points_1 = int(obs['player_1'].team_points[1])
        previous_obs = obs

    elif reward['player_1'] > reward_memory['player_1'] :
        reward_memory = reward
        points_0 = int(obs['player_0'].team_points[0])
        points_1 = int(obs['player_1'].team_points[1])
        previous_obs = obs

    else :
        points_0 = int(obs['player_0'].team_points[0] - previous_obs['player_0'].team_points[0])
        points_1 = int(obs['player_1'].team_points[1] - previous_obs['player_1'].team_points[1])
        previous_obs = obs

    print(points_0,points_1)

    state_maps_0 ,state_features_0 = policy_0.obs_to_state(obs['player_0'],params,points_0,state_maps_0,show=show)
    action_0, value, mask_action, mask_dx, mask_dy, log_prob = policy_0(state_maps_0 ,state_features_0,obs['player_0'],params)

    state_maps_1 ,state_features_1 = policy_1.obs_to_state(obs['player_1'],params,points_1,state_maps_1,show=show)
    action_1, value, mask_action, mask_dx, mask_dy, log_prob = policy_1(state_maps_1 ,state_features_1,obs['player_1'],params)

    action = dict(player_0=np.array(action_0, dtype=int), player_1=np.array(action_1, dtype=int))

    rng, key_step = jax.random.split(rng,2)
    obs, state, reward, terminated, truncated, info = env.step(key_step,state,action,params)

    


   