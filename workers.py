from src.luxai_s3.wrappers import LuxAIS3GymEnv
import random 
import torch
import torch.multiprocessing as mp
import numpy as np


class Luxai_Worker(mp.Process) :

    def __init__(self, worker_id, shared_queue, policy_0, policy_1, victory_bonus, gamma, gae_lambda, n_steps,reward_queue, event) :

        super(Luxai_Worker, self).__init__()

        self.env = LuxAIS3GymEnv(numpy_output=True)
        self.n_units = self.env.env_params.max_units
        self.max_unit_energy = self.env.env_params.max_unit_energy
        self.map_width = self.env.env_params.map_width
        self.map_height = self.env.env_params.map_height
        self.max_relic_nodes = self.env.env_params.max_relic_nodes     
        self.match_step = self.env.env_params.max_steps_in_match + 1
        self.len_episode = self.match_step * self.env.env_params.match_count_per_episode
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_steps = n_steps

        self.sap_range = 8

        self.victory_bonus = victory_bonus

        self.policy_0 = policy_0
        self.policy_1 = policy_1

        ep_params = {}
        ep_params['unit_move_cost'] = self.env.env_params.unit_move_cost
        ep_params['unit_sap_cost'] = self.env.env_params.unit_sap_cost
        ep_params['unit_sap_range'] = self.env.env_params.unit_sap_range
        ep_params['unit_sensor_range'] = self.env.env_params.unit_sensor_range

        obs, _ = self.env.reset()
        state_maps, state_features = self.policy_0.obs_to_state(obs['player_0'],ep_params)
        self.n_inputs_features = state_features.size(0)
        self.n_inputs_maps = state_maps.size(0)
        self.n_action = 6

        self.shared_queue = shared_queue
        self.reward_queue = reward_queue
        self.event = event
        self.worker_id = worker_id

    def random_seed(self,length):
        random.seed()
        min = 10**(length-1)
        max = 9*min + (min-1)
        return random.randint(min, max)
    
    def run(self) :

        with torch.no_grad():
            not_begin_episode = True

            states_features = torch.zeros(2,self.n_steps,self.n_inputs_features,dtype=torch.float)
            states_maps = torch.zeros(2,self.n_steps,self.n_inputs_maps,self.map_width,self.map_height,dtype=torch.float)
            actions = torch.zeros(2,self.n_steps,self.n_units,3,dtype=torch.int)
            values = torch.zeros(2,self.n_steps,1,dtype=torch.float)
            rewards = torch.zeros(2,self.n_steps,dtype=torch.float)
            episode_start = torch.zeros(self.n_steps,dtype=torch.float)
            mask_actions = torch.zeros(2,self.n_steps,self.n_units,self.n_action,dtype=torch.int8)
            mask_dxs = torch.zeros(2,self.n_steps,self.n_units,self.sap_range*2+1,dtype=torch.int8)
            mask_dys = torch.zeros(2,self.n_steps,self.n_units,self.sap_range*2+1,dtype=torch.int8)

            step_cpt = 0

            while True:

                # Reset the environment and get the initial state
                obs, _ = self.env.reset(seed=self.random_seed(10))

                if not_begin_episode :
                    for i in range(self.worker_id*62) :
                        action = dict(
                                    player_0=np.random.randint(0,4,size=(self.n_units, 3)),
                                    player_1=np.random.randint(0,4,size=(self.n_units, 3))
                                )
                        _, _,  _, _ , _ = self.env.step(action)
                    not_begin_episode = False

                ep_params = {}
                ep_params['unit_move_cost'] = self.env.env_params.unit_move_cost
                ep_params['unit_sap_cost'] = self.env.env_params.unit_sap_cost
                ep_params['unit_sap_range'] = self.env.env_params.unit_sap_range
                ep_params['unit_sensor_range'] = self.env.env_params.unit_sensor_range

                state_maps_0, state_features_0 = self.policy_0.obs_to_state(obs['player_0'],ep_params)
                state_maps_1, state_features_1 = self.policy_1.obs_to_state(obs['player_1'],ep_params)
                previous_obs = obs
                
                cumulated_reward = torch.zeros(2,dtype=torch.float)

                map_0 = torch.zeros(self.map_width,self.map_height,dtype=torch.float)
                map_1 = torch.zeros(self.map_width,self.map_height,dtype=torch.float)

                energy_0 = torch.zeros(self.n_units,1,dtype=torch.float)
                energy_1 = torch.zeros(self.n_units,1,dtype=torch.float)

                enemy_0 = torch.zeros(self.n_units,1,dtype=torch.float)
                enemy_1 = torch.zeros(self.n_units,1,dtype=torch.float)

                for ep_step in range(self.len_episode):

                    #Compute action probabilities with masks and sample action
                    action_0 , value_0, mask_action_0 , mask_dx_0, mask_dy_0  = self.policy_0(state_maps_0,state_features_0,obs['player_0'],ep_params)
                    action_1 , value_1, mask_action_1 , mask_dx_1, mask_dy_1  = self.policy_1(state_maps_1,state_features_1,obs['player_1'],ep_params)

                    # Take a step in the environment
                    action = dict(player_0=np.array(action_0, dtype=int), player_1=np.array(action_1, dtype=int))
                    obs, reward, truncated, done, info = self.env.step(action)

                    # If the Buffer is full, send collected trajectories to the Queue
                    if step_cpt == self.n_steps :
                        
                        # Advantage computation
                        advantages = torch.zeros(2,self.n_steps,1,dtype=torch.float)                    
                        last_gae_lam_0 = 0
                        last_gae_lam_1 = 0

                        for step in reversed(range(self.n_steps)):

                            if step == self.n_steps - 1 :
                                next_values_0 = value_0
                                next_values_1 = value_1
                                non_terminal = 1 - float(ep_step==0)

                            else:
                                next_values_0 = values[0,step + 1]
                                next_values_1 = values[1,step + 1]
                                non_terminal = 1 - episode_start[step + 1]

                            delta_0 = rewards[0,step] + self.gamma * next_values_0 * non_terminal - values[0,step]
                            delta_1 = rewards[1,step] + self.gamma * next_values_1 * non_terminal - values[1,step]

                            last_gae_lam_0 = delta_0 + self.gamma * self.gae_lambda * non_terminal * last_gae_lam_0
                            last_gae_lam_1 = delta_1 + self.gamma * self.gae_lambda * non_terminal * last_gae_lam_1

                            advantages[0,step] = last_gae_lam_0
                            advantages[1,step] = last_gae_lam_1

                        returns = advantages + values

                        self.shared_queue.put((states_maps,states_features,actions,advantages,returns,mask_actions,mask_dxs,mask_dys))

                        step_cpt = 0
                        states_features = torch.zeros(2,self.n_steps,self.n_inputs_features,dtype=torch.float)
                        states_maps = torch.zeros(2,self.n_steps,self.n_inputs_maps,self.map_width,self.map_height,dtype=torch.float)
                        actions = torch.zeros(2,self.n_steps,self.n_units,3,dtype=torch.int)
                        values = torch.zeros(2,self.n_steps,1,dtype=torch.float)
                        rewards = torch.zeros(2,self.n_steps,dtype=torch.float)
                        episode_start = torch.zeros(self.n_steps,dtype=torch.float)
                        mask_actions = torch.zeros(2,self.n_steps,self.n_units,self.n_action,dtype=torch.int8)
                        mask_dxs = torch.zeros(2,self.n_steps,self.n_units,self.sap_range*2+1,dtype=torch.int8)
                        mask_dys = torch.zeros(2,self.n_steps,self.n_units,self.sap_range*2+1,dtype=torch.int8)

                        self.event.wait()
                        self.event.clear()

                    # Compute the rewards
                    next_state_maps_0, next_state_features_0 = self.policy_0.obs_to_state(obs['player_0'],ep_params)
                    next_state_maps_1, next_state_features_1 = self.policy_1.obs_to_state(obs['player_1'],ep_params)
                    episode_start[step_cpt] = 0

                    non_reset_matchs = True
                    
                    if ep_step == 0 :
                        episode_start[step_cpt] = 1
                        reward_memory = reward
                        rewards[0,step_cpt] += obs['player_0']['team_points'][0] / 500
                        rewards[1,step_cpt] += obs['player_1']['team_points'][1] / 500
                        non_reset_matchs = False

                    elif reward['player_0'] > reward_memory['player_0'] :
                        reward_memory = reward
                        rewards[0,step_cpt] += (obs['player_0']['team_points'][0] + self.victory_bonus) / 500
                        rewards[1,step_cpt] += obs['player_1']['team_points'][1] / 500
                        non_reset_matchs = False

                    elif reward['player_1'] > reward_memory['player_1'] :
                        reward_memory = reward
                        rewards[0,step_cpt] += obs['player_0']['team_points'][0] / 500
                        rewards[1,step_cpt] += (obs['player_1']['team_points'][1] + self.victory_bonus) / 500
                        non_reset_matchs = False

                    else :
                        a=0
                        rewards[0,step_cpt] += (obs['player_0']['team_points'][0] - previous_obs['player_0']['team_points'][0]) / 500
                        rewards[1,step_cpt] += (obs['player_1']['team_points'][1] - previous_obs['player_1']['team_points'][1]) / 500

                    new_map_0 = torch.clamp_max(map_0 + torch.from_numpy(obs['player_0']['sensor_mask'].astype(np.float32)),1)
                    new_map_1 = torch.clamp_max(map_1 + torch.from_numpy(obs['player_1']['sensor_mask'].astype(np.float32)),1)
                    new_energy_0 = torch.from_numpy(obs['player_0']['units']['energy'][0].astype(np.float32))
                    new_energy_1 = torch.from_numpy(obs['player_1']['units']['energy'][1].astype(np.float32))
                    new_enemy_0 = torch.from_numpy(obs['player_0']['units_mask'][1].astype(np.float32))
                    new_enemy_1 = torch.from_numpy(obs['player_1']['units_mask'][0].astype(np.float32))

                    if non_reset_matchs :
                        rewards[0,step_cpt] += torch.sum(new_map_0-map_0) / (self.map_width*self.map_height)
                        rewards[1,step_cpt] += torch.sum(new_map_1-map_1) / (self.map_width*self.map_height)

                        rewards[0,step_cpt] += torch.sum(new_energy_0-energy_0) / (self.max_unit_energy*self.n_units)
                        rewards[1,step_cpt] += torch.sum(new_energy_1-energy_1) / (self.max_unit_energy*self.n_units)

                        rewards[0,step_cpt] -= torch.sum(new_energy_1*new_enemy_0 - energy_1*enemy_0) / (self.max_unit_energy*self.n_units)
                        rewards[1,step_cpt] -= torch.sum(new_energy_0*new_enemy_1 - energy_0*enemy_1) / (self.max_unit_energy*self.n_units)

                        rewards[0,step_cpt] += torch.sum(torch.from_numpy(obs['player_0']['relic_nodes_mask'].astype(np.float32))) / (self.max_relic_nodes*50)
                        rewards[1,step_cpt] += torch.sum(torch.from_numpy(obs['player_0']['relic_nodes_mask'].astype(np.float32))) / (self.max_relic_nodes*50)

                    else :
                        map_0 = torch.zeros(self.map_width,self.map_height,dtype=torch.float)
                        map_1 = torch.zeros(self.map_width,self.map_height,dtype=torch.float)

                    map_0 = new_map_0
                    map_1 = new_map_1
                    energy_0 = new_energy_0
                    energy_1 = new_energy_1
                    enemy_0 = new_enemy_0
                    enemy_1 = new_enemy_1
                    
                    cumulated_reward[0] += rewards[0,step_cpt]
                    cumulated_reward[1] += rewards[1,step_cpt]

                    #Update the trajectories
                    states_maps[0,step_cpt] = state_maps_0
                    states_features[0,step_cpt] = state_features_0
                    states_maps[1,step_cpt] = state_maps_1
                    states_features[1,step_cpt] = state_features_1

                    actions[0,step_cpt] = action_0
                    actions[1,step_cpt] = action_1

                    values[0,step_cpt] = value_0
                    values[1,step_cpt] = value_1

                    mask_actions[0,step_cpt] = mask_action_0
                    mask_actions[1,step_cpt] = mask_action_1

                    mask_dxs[0,step_cpt] = mask_dx_0
                    mask_dxs[1,step_cpt] = mask_dx_1
                    mask_dys[0,step_cpt] = mask_dy_0
                    mask_dys[1,step_cpt] = mask_dy_1

                    state_maps_0 = next_state_maps_0
                    state_features_0 = next_state_features_0
                    state_maps_1 = next_state_maps_1
                    state_features_1 = next_state_features_1

                    step_cpt += 1
                    previous_obs = obs

                self.reward_queue.put(cumulated_reward)
