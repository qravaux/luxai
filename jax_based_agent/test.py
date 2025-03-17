import sys
sys.path.append('../')
from src.luxai_s3.env import LuxAIS3Env
from src.luxai_s3.params import EnvParams, env_params_ranges
from kits.python.lux.kit import to_jax
import numpy as np
from jax_policies import jax_Luxai_Agent
from jax_utils import *
import torch
import numpy as np
import jax
import time
import matplotlib.pyplot as plt
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
rng = jax.random.key(5)
rng, key_reset = jax.random.split(rng, 2)
params = random_params(key_reset)
obs, state = env.reset(key_reset,params)

target_dist_0, target_0 = compute_target_distance_matrix(state,obs,'player_0')
old_distance_0 = compute_distance(obs,target_dist_0,'player_0')
target_dist_1, target_1 = compute_target_distance_matrix(state,obs,'player_1')
old_distance_1 = compute_distance(obs,target_dist_1,'player_1')

state_maps_0 = jnp.zeros((10, 24, 24), dtype=jnp.float32)
state_maps_0 = state_maps_0.at[4].set(-1) / 20
state_maps_0 = state_maps_0.at[5].set(-1)
state_maps_0 = state_maps_0.at[9].set(-1)

points_0 = jnp.zeros(1, dtype=jnp.float32)

state_maps_1 = jnp.zeros((10, 24, 24), dtype=jnp.float32)
state_maps_1 = state_maps_1.at[4].set(-1) / 20
state_maps_1 = state_maps_1.at[5].set(-1)
state_maps_1 = state_maps_1.at[9].set(-1)

points_1 = jnp.zeros(1, dtype=jnp.float32)
previous_obs = obs

policy = jax_Luxai_Agent()

policy.load_state_dict(torch.load("policy/corrected_state_1e-16/policy_step_75.pth", weights_only=True))

for step in range(505) :

    state_maps_0 ,state_features_0 = obs_to_state(obs,params,points_0,state_maps_0,'player_0')
    state_maps_1 ,state_features_1 = obs_to_state(obs,params,points_1,state_maps_1,'player_1')

    
    if step%10 == 0:
        renderer.render(state,params)
        """
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        axes = axes.flatten()
        for i in range(10):
            axes[i].imshow(state_maps_0[i].T)
        plt.show()
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        axes = axes.flatten()
        for i in range(10):
            axes[i].imshow(state_maps_1[i].T)
        plt.show()"
        """

    actor_action_0, actor_dx_0, actor_dy_0, value_0 = policy(jnp.expand_dims(state_maps_0,axis=0),jnp.expand_dims(state_features_0,axis=0),"cpu")
    rng, action_key = jax.random.split(rng,2)
    action_0, mask_action, mask_dx, mask_dy, log_prob = compute_mask_actions_log_probs(obs,params,action_key,actor_action_0[0],actor_dx_0[0], actor_dy_0[0],'player_0')

    actor_action_1, actor_dx_1, actor_dy_1, value_1 = policy(jnp.expand_dims(state_maps_1,axis=0),jnp.expand_dims(state_features_1,axis=0),"cpu")
    rng, action_key = jax.random.split(rng,2)
    action_1, mask_action, mask_dx, mask_dy, log_prob = compute_mask_actions_log_probs(obs,params,action_key,actor_action_1[0],actor_dx_1[0], actor_dy_1[0],'player_1')

    action = dict(player_0=action_0, player_1=swap_action(action_1))

    rng, key_step = jax.random.split(rng,2)
    obs, state, reward, terminated, truncated, info = env.step(key_step,state,action,params)

    target_dist_0, target_0 = compute_target_distance_matrix(state,obs,'player_0')
    distance_0 = compute_distance(obs,target_dist_0,'player_0')
    reward_0, units_mask = compute_reward(obs,previous_obs,params,action_0,distance_0,old_distance_0,target_0,'player_0')
    points_0, points_1 = compute_points(obs,previous_obs)

    target_dist_1, target_1 = compute_target_distance_matrix(state,obs,'player_1')
    distance_1 = compute_distance(obs,target_dist_1,'player_1')
    reward_1, units_mask = compute_reward(obs,previous_obs,params,action_1,distance_1,old_distance_1,target_1,'player_1')

    previous_obs = obs        
    old_distance_0 = distance_0
    old_distance_1 = distance_1
    

    


   