import sys
sys.path.append('../')
import torch.multiprocessing as mp
from jax_utils import *
from src.luxai_s3.env import LuxAIS3Env
import jax
import jax.numpy as jnp
import time
import torch

class jax_Luxai_Worker(mp.Process) :

    def __init__(self,
                 worker_id,
                 policy,
                 queue,
                 reward_queue,
                 point_queue,
                 num_envs,
                 num_workers,
                 gamma, 
                 gae_lambda, 
                 n_steps,
                 n_episode,
                 file_name,
                 device) :

        super(jax_Luxai_Worker, self).__init__()

        self.worker_id = worker_id
        self.policy = policy
        self.queue = queue
        self.reward_queue = reward_queue
        self.point_queue = point_queue
        self.num_envs = num_envs
        self.num_workers = num_workers
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_episode = n_episode
        self.n_steps = n_steps
        self.file_name = file_name
        self.device = device

    def run(self) :

        #Create env and vmap functions
        env = LuxAIS3Env(auto_reset=False)
        vmap_random_gen = jax.vmap(random_params, in_axes=0)
        vmap_reset = jax.vmap(env.reset, in_axes=(0, 0))
        vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, 0))
        vmap_obs_to_state = jax.vmap(obs_to_state,in_axes=(0,0,0,0,None))
        vmap_compute_mask_actions_log_probs = jax.vmap(compute_mask_actions_log_probs, in_axes=(0,0,0,0,0,0,None))
        vmap_compute_points = jax.vmap(compute_points, in_axes=(0,0))
        vmap_compute_target_distance_matrix = jax.vmap(compute_target_distance_matrix, in_axes=(0,0,None))
        vmap_compute_distance = jax.vmap(compute_distance, in_axes=(0,0,None))
        vmap_compute_reward = jax.vmap(compute_reward,in_axes=(0,0,0,0,0,0,0,None))
        vmap_swap_action = jax.vmap(swap_action,in_axes=0)
        vmap_generate_random_action = jax.vmap(generate_random_action,in_axes=0)

        #Create rng keys and counters
        rng = jax.random.key(0)
        rng, key_reset, key_step = jax.random.split(rng, 3)

        steps_cpt = 0
        queue_cpt = 0
        not_begin_episode = True

        #Initialise jnp arrays for policy feeding
        n_inputs_features = 156
        n_inputs_maps = 10
        map_width = 24
        map_height = 24
        n_units = 16
        n_action = 6
        sap_range = 8

        b_state_maps = jnp.zeros((self.n_steps,2,self.num_envs,n_inputs_maps,map_width,map_height),dtype=jnp.float32)
        b_state_features = jnp.zeros((self.n_steps,2,self.num_envs,n_inputs_features),dtype=jnp.float32)
        b_action = jnp.zeros((self.n_steps,2,self.num_envs,n_units,3),dtype=jnp.int32)
        b_value = jnp.zeros((self.n_steps,2,self.num_envs,n_units),dtype=jnp.float32)
        b_reward = jnp.zeros((self.n_steps,2,self.num_envs,n_units),dtype=jnp.float32)
        b_log_prob = jnp.zeros((self.n_steps,2,self.num_envs,n_units),dtype=jnp.float32)
        b_mask_action = jnp.zeros((self.n_steps,2,self.num_envs,n_units,n_action),dtype=jnp.float32)
        b_mask_dx = jnp.zeros((self.n_steps,2,self.num_envs,n_units,sap_range*2+1),dtype=jnp.float32)
        b_mask_dy = jnp.zeros((self.n_steps,2,self.num_envs,n_units,sap_range*2+1),dtype=jnp.float32)
        b_mask_unit = jnp.zeros((self.n_steps,2,self.num_envs,n_units),dtype=jnp.float32)
        b_episode_start = jnp.zeros((self.n_steps,2,self.num_envs,n_units),dtype=jnp.float32)

        start = time.time()
        for episode in range(self.n_episode):

            #Generate random env params
            vmap_key_params = jax.random.split(key_reset, self.num_envs)
            env_params = vmap_random_gen(vmap_key_params)

            #Reset environment, map memory and points
            vmap_key_reset = jax.random.split(key_reset, self.num_envs)
            obs, state = vmap_reset(vmap_key_reset, env_params)
            state_maps_0, points_0 = generate_map_memory(self.num_envs)
            state_maps_1, points_1 = generate_map_memory(self.num_envs)
            previous_obs = obs

            target_dist_0, target_0 = vmap_compute_target_distance_matrix(state,obs,'player_0')
            old_distance_0 = vmap_compute_distance(obs,target_dist_0,'player_0')

            target_dist_1, target_1 = vmap_compute_target_distance_matrix(state,obs,'player_1')
            old_distance_1 = vmap_compute_distance(obs,target_dist_1,'player_1')

            cumulated_rewards = jnp.zeros((2,self.num_envs))
            cumulated_points = jnp.zeros((2,self.num_envs))

            for step in range(505) :

                if not_begin_episode and self.worker_id > 0 :
                    vmap_keys_step = jax.random.split(key_step, self.num_envs)
                    action = vmap_generate_random_action(vmap_keys_step)
                    obs, state, reward, terminated, truncated, info = vmap_step(vmap_keys_step, state, action, env_params)
                    
                    points_0, points_1 = vmap_compute_points(obs,previous_obs)

                    target_dist_0, target_0 = vmap_compute_target_distance_matrix(state,obs,'player_0')
                    distance_0 = vmap_compute_distance(obs,target_dist_0,'player_0')
                    reward_0 = vmap_compute_reward(obs,previous_obs,action['player_0'],distance_0,old_distance_0,target_0,'player_0')

                    target_dist_1, target_1 = vmap_compute_target_distance_matrix(state,obs,'player_1')
                    distance_1 = vmap_compute_distance(obs,target_dist_1,'player_1')
                    reward_1 = vmap_compute_reward(obs,previous_obs,action['player_1'],distance_1,old_distance_1,target_1,'player_1')

                    cumulated_rewards = cumulated_rewards.at[0].add(jnp.mean(reward_0,axis=-1))
                    cumulated_rewards = cumulated_rewards.at[1].add(jnp.mean(reward_1,axis=-1))
                    cumulated_points = cumulated_points.at[0].add(points_0)
                    cumulated_points = cumulated_points.at[1].add(points_1)

                    previous_obs = obs
                    old_distance_0 = distance_0
                    old_distance_1 = distance_1
                    not_begin_episode = ((step+1)<(self.worker_id*self.n_steps))
                    
                    if not not_begin_episode :
                        print(f"worker {self.worker_id} starts at step {step+2}")
                    continue

                episode_start = jnp.ones((self.num_envs,n_units)) * int(step==0)
                points_0, points_1 = vmap_compute_points(obs,previous_obs)

                #Generate new step keys
                vmap_keys_step = jax.random.split(key_step, self.num_envs)
                #Compute state and action for player 0
                state_maps_0, state_features_0 = vmap_obs_to_state(obs, env_params, points_0, state_maps_0, 'player_0')
                actor_action_0, actor_dx_0, actor_dy_0, value_0 = self.policy(state_maps_0, state_features_0, self.device)
                action_0, mask_action_0, mask_dx_0, mask_dy_0, log_prob_0 = vmap_compute_mask_actions_log_probs(obs,
                                                                                                                env_params,
                                                                                                                vmap_keys_step,
                                                                                                                actor_action_0,
                                                                                                                actor_dx_0,
                                                                                                                actor_dy_0,
                                                                                                                'player_0') 
                
                #Generate new step keys
                vmap_keys_step = jax.random.split(key_step, self.num_envs)
                #Compute state and action for player 1
                state_maps_1, state_features_1 = vmap_obs_to_state(obs, env_params, points_1, state_maps_1, 'player_1')
                actor_action_1, actor_dx_1, actor_dy_1, value_1 = self.policy(state_maps_1, state_features_1, self.device)
                action_1, mask_action_1, mask_dx_1, mask_dy_1, log_prob_1 = vmap_compute_mask_actions_log_probs(obs,
                                                                                                                env_params,
                                                                                                                vmap_keys_step,
                                                                                                                actor_action_1,
                                                                                                                actor_dx_1,
                                                                                                                actor_dy_1,
                                                                                                                'player_1')
                
                #Gather action from both players and computing next step
                action = dict(player_0=action_0,player_1=vmap_swap_action(action_1))
                obs, state, reward, terminated, truncated, info = vmap_step(vmap_keys_step, state, action, env_params)

                target_dist_0, target_0 = vmap_compute_target_distance_matrix(state,obs,'player_0')
                distance_0 = vmap_compute_distance(obs,target_dist_0,'player_0')
                reward_0, units_mask_0 = vmap_compute_reward(obs,previous_obs,env_params,action_0,distance_0,old_distance_0,target_0,'player_0')

                target_dist_1, target_1 = vmap_compute_target_distance_matrix(state,obs,'player_1')
                distance_1 = vmap_compute_distance(obs,target_dist_1,'player_1')
                reward_1, units_mask_1 = vmap_compute_reward(obs,previous_obs,env_params,action_1,distance_1,old_distance_1,target_1,'player_1')

                cumulated_rewards = cumulated_rewards.at[0].add(jnp.mean(reward_0,axis=-1))
                cumulated_rewards = cumulated_rewards.at[1].add(jnp.mean(reward_1,axis=-1))
                cumulated_points = cumulated_points.at[0].add(points_0)
                cumulated_points = cumulated_points.at[1].add(points_1)

                previous_obs = obs
                old_distance_0 = distance_0
                old_distance_1 = distance_1

                if steps_cpt == self.n_steps :

                    queue_cpt += 1

                    b_advantage = compute_advantage(b_value,b_episode_start,b_reward,episode_start,jnp.stack([value_0,value_1],axis=0),self.gamma,self.gae_lambda,self.n_steps,self.num_envs)

                    b_return = b_advantage + b_value

                    s_m = torch.tensor(b_state_maps).flatten(0,2)
                    s_f = torch.tensor(b_state_features).flatten(0,2)
                    ac = torch.tensor(b_action).flatten(0,2)
                    ad = torch.tensor(b_advantage).flatten(0,2)
                    re = torch.tensor(b_return).flatten(0,2)
                    lo = torch.tensor(b_log_prob).flatten(0,2)
                    m_a = torch.tensor(b_mask_action).flatten(0,2)
                    m_x = torch.tensor(b_mask_dx).flatten(0,2)
                    m_y = torch.tensor(b_mask_dy).flatten(0,2)
                    u_m = torch.tensor(b_mask_unit).flatten(0,2)
                    
                    print(f'worker {self.worker_id} complete step n°{queue_cpt} at speed of {round((self.n_steps*self.num_envs)/(time.time()-start),1)}it/s')
                    self.queue.put((s_m,s_f,ac,ad,re,lo,m_a,m_x,m_y,u_m))
                    start = time.time()

                    steps_cpt = 0
                    b_state_maps = jnp.zeros((self.n_steps,2,self.num_envs,n_inputs_maps,map_width,map_height),dtype=jnp.float32)
                    b_state_features = jnp.zeros((self.n_steps,2,self.num_envs,n_inputs_features),dtype=jnp.float32)
                    b_action = jnp.zeros((self.n_steps,2,self.num_envs,n_units,3),dtype=jnp.int32)
                    b_value = jnp.zeros((self.n_steps,2,self.num_envs,n_units),dtype=jnp.float32)
                    b_reward = jnp.zeros((self.n_steps,2,self.num_envs,n_units),dtype=jnp.float32)
                    b_log_prob = jnp.zeros((self.n_steps,2,self.num_envs,n_units),dtype=jnp.float32)
                    b_mask_action = jnp.zeros((self.n_steps,2,self.num_envs,n_units,n_action),dtype=jnp.float32)
                    b_mask_dx = jnp.zeros((self.n_steps,2,self.num_envs,n_units,sap_range*2+1),dtype=jnp.float32)
                    b_mask_dy = jnp.zeros((self.n_steps,2,self.num_envs,n_units,sap_range*2+1),dtype=jnp.float32)
                    b_mask_unit = jnp.zeros((self.n_steps,2,self.num_envs,n_units),dtype=jnp.float32)
                    b_episode_start = jnp.zeros((self.n_steps,2,self.num_envs,n_units),dtype=jnp.float32)
                
                b_state_maps = b_state_maps.at[steps_cpt,0].set(state_maps_0)
                b_state_features = b_state_features.at[steps_cpt,0].set(state_features_0)
                b_action = b_action.at[steps_cpt,0].set(action_0)
                b_value = b_value.at[steps_cpt,0].set(value_0)
                b_reward = b_reward.at[steps_cpt,0].set(reward_0)
                b_log_prob = b_log_prob.at[steps_cpt,0].set(log_prob_0)
                b_mask_action = b_mask_action.at[steps_cpt,0].set(mask_action_0)
                b_mask_dx = b_mask_dx.at[steps_cpt,0].set(mask_dx_0)
                b_mask_dy = b_mask_dy.at[steps_cpt,0].set(mask_dy_0)
                b_mask_unit = b_mask_unit.at[steps_cpt,0].set(units_mask_0)
                b_episode_start = b_episode_start.at[steps_cpt,0].set(episode_start)

                b_state_maps = b_state_maps.at[steps_cpt,1].set(state_maps_1)
                b_state_features = b_state_features.at[steps_cpt,1].set(state_features_1)
                b_action = b_action.at[steps_cpt,1].set(action_1)
                b_value = b_value.at[steps_cpt,1].set(value_1)
                b_reward = b_reward.at[steps_cpt,1].set(reward_1)
                b_log_prob = b_log_prob.at[steps_cpt,1].set(log_prob_1)
                b_mask_action = b_mask_action.at[steps_cpt,1].set(mask_action_1)
                b_mask_dx = b_mask_dx.at[steps_cpt,1].set(mask_dx_1)
                b_mask_dy = b_mask_dy.at[steps_cpt,1].set(mask_dy_1)
                b_mask_unit = b_mask_unit.at[steps_cpt,1].set(units_mask_1)
                b_episode_start = b_episode_start.at[steps_cpt,1].set(episode_start)

                steps_cpt += 1
            
            cumulated_rewards = torch.tensor(cumulated_rewards)
            cumulated_points = torch.tensor(cumulated_points)

            self.reward_queue.put(cumulated_rewards)
            self.point_queue.put(cumulated_points)

        self.writer.close()
