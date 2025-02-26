from src.luxai_s3.wrappers import LuxAIS3GymEnv
import random 
import torch
import torch.multiprocessing as mp
import numpy as np
import time

class Luxai_Worker(mp.Process) :

    def __init__(self, 
                 worker_id, 
                 shared_queue,
                 reward_queue,
                 point_queue,
                 event,
                 policy_0, 
                 policy_1,
                 gamma, 
                 gae_lambda, 
                 n_steps,
                 n_episode,
                 n_epochs,
                 victory_bonus,
                 ) :

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
        self.n_episode = n_episode
        self.n_epochs = n_epochs

        self.sap_range = 8

        self.victory_bonus = victory_bonus

        self.policy_0 = policy_0
        self.policy_1 = policy_1

        self.ep_params = {}
        self.ep_params['unit_move_cost'] = self.env.env_params.unit_move_cost
        self.ep_params['unit_sap_cost'] = self.env.env_params.unit_sap_cost
        self.ep_params['unit_sap_range'] = self.env.env_params.unit_sap_range
        self.ep_params['unit_sensor_range'] = self.env.env_params.unit_sensor_range

        obs, _ = self.env.reset()
        state_maps, state_features = self.policy_0.obs_to_state(obs['player_0'],self.ep_params)
        self.n_inputs_features = state_features.size(0)
        self.n_inputs_maps = state_maps.size(0)
        self.n_action = 6

        self.shared_queue = shared_queue
        self.reward_queue = reward_queue
        self.point_queue = point_queue
        self.event = event
        self.worker_id = worker_id

    def random_seed(self,length):
        #return(20)
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
            values = torch.zeros(2,self.n_steps,self.n_units,dtype=torch.float)
            rewards = torch.zeros(2,self.n_steps,self.n_units,dtype=torch.float)
            episode_start = torch.zeros(self.n_steps,dtype=torch.float)
            mask_actions = torch.zeros(2,self.n_steps,self.n_units,self.n_action,dtype=torch.int8)
            mask_dxs = torch.zeros(2,self.n_steps,self.n_units,self.sap_range*2+1,dtype=torch.int8)
            mask_dys = torch.zeros(2,self.n_steps,self.n_units,self.sap_range*2+1,dtype=torch.int8)
            log_probs = torch.zeros(2,self.n_steps,self.n_units,dtype=torch.float)

            step_cpt = 0
            episode_cpt = 0
            global_count = 0

            while global_count < self.n_epochs*self.n_episode :

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

                self.ep_params = {}
                self.ep_params['unit_move_cost'] = self.env.env_params.unit_move_cost
                self.ep_params['unit_sap_cost'] = self.env.env_params.unit_sap_cost
                self.ep_params['unit_sap_range'] = self.env.env_params.unit_sap_range
                self.ep_params['unit_sensor_range'] = self.env.env_params.unit_sensor_range

                state_maps_0, state_features_0 = self.policy_0.obs_to_state(obs['player_0'],self.ep_params)
                state_maps_1, state_features_1 = self.policy_1.obs_to_state(obs['player_1'],self.ep_params)

                previous_obs = obs
                
                cumulated_reward = torch.zeros(2,dtype=torch.float)
                cumulated_point = torch.zeros(2,dtype=torch.float)

                energy_0 = torch.zeros(self.n_units,dtype=torch.float)
                energy_1 = torch.zeros(self.n_units,dtype=torch.float)

                distance_explore_0 = torch.zeros(self.n_units,dtype=torch.float)
                distance_explore_1 = torch.zeros(self.n_units,dtype=torch.float)


                for ep_step in range(self.len_episode):

                    #Compute action probabilities with masks and sample action
                    action_0 , value_0, mask_action_0 , mask_dx_0, mask_dy_0, log_prob_0 = self.policy_0(state_maps_0,state_features_0,obs['player_0'],self.ep_params)
                    action_1 , value_1, mask_action_1 , mask_dx_1, mask_dy_1, log_prob_1 = self.policy_1(state_maps_1,state_features_1,obs['player_1'],self.ep_params)

                    # Take a step in the environment
                    action = dict(player_0=np.array(action_0, dtype=int), player_1=np.array(action_1, dtype=int))
                    obs, reward, truncated, done, info = self.env.step(action)

                    # If the Buffer is full, send collected trajectories to the Queue
                    if step_cpt == self.n_steps :

                        # Advantage computation
                        advantages = torch.zeros(2,self.n_steps,self.n_units,dtype=torch.float)                    
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

                        self.shared_queue.put((states_maps,states_features,actions,advantages,returns,mask_actions,mask_dxs,mask_dys,log_probs))

                        step_cpt = 0
                        states_features = torch.zeros(2,self.n_steps,self.n_inputs_features,dtype=torch.float)
                        states_maps = torch.zeros(2,self.n_steps,self.n_inputs_maps,self.map_width,self.map_height,dtype=torch.float)
                        actions = torch.zeros(2,self.n_steps,self.n_units,3,dtype=torch.int)
                        values = torch.zeros(2,self.n_steps,self.n_units,dtype=torch.float)
                        rewards = torch.zeros(2,self.n_steps,self.n_units,dtype=torch.float)
                        episode_start = torch.zeros(self.n_steps,dtype=torch.float)
                        mask_actions = torch.zeros(2,self.n_steps,self.n_units,self.n_action,dtype=torch.int8)
                        mask_dxs = torch.zeros(2,self.n_steps,self.n_units,self.sap_range*2+1,dtype=torch.int8)
                        mask_dys = torch.zeros(2,self.n_steps,self.n_units,self.sap_range*2+1,dtype=torch.int8)
                        log_probs = torch.zeros(2,self.n_steps,self.n_units,dtype=torch.float)
                        
                        global_count += 1
                        episode_cpt += 1
                        if episode_cpt == self.n_episode and global_count < self.n_epochs*self.n_episode :
                            self.event.wait()
                            self.event.clear()
                            episode_cpt = 0

                    # Compute the rewards

                    episode_start[step_cpt] = 0

                    if ep_step == 0 :
                        episode_start[step_cpt] = 1
                        reward_memory = reward
                        points_0 = obs['player_0']['team_points'][0]
                        points_1 = obs['player_1']['team_points'][1]
                        non_reset_matchs = True

                    elif reward['player_0'] > reward_memory['player_0'] :
                        reward_memory = reward
                        points_0 = (obs['player_0']['team_points'][0] + self.victory_bonus)
                        points_1 = obs['player_1']['team_points'][1]
                        non_reset_matchs = False

                    elif reward['player_1'] > reward_memory['player_1'] :
                        reward_memory = reward
                        points_0 = obs['player_0']['team_points'][0]
                        points_1 = (obs['player_1']['team_points'][1] + self.victory_bonus)
                        non_reset_matchs = False

                    else :
                        non_reset_matchs = True
                        points_0 = (obs['player_0']['team_points'][0] - previous_obs['player_0']['team_points'][0])
                        points_1 = (obs['player_1']['team_points'][1] - previous_obs['player_1']['team_points'][1])

                    next_state_maps_0, next_state_features_0 = self.policy_0.obs_to_state(obs['player_0'],self.ep_params,points_0,state_maps_0)
                    next_state_maps_1, next_state_features_1 = self.policy_1.obs_to_state(obs['player_1'],self.ep_params,points_1,state_maps_1)

                    cumulated_point[0] += points_0
                    cumulated_point[1] += points_1

                    new_energy_0 = torch.from_numpy(obs['player_0']['units']['energy'][0].astype(np.int32)).view(self.n_units)
                    new_energy_1 = torch.from_numpy(obs['player_1']['units']['energy'][1].astype(np.int32)).view(self.n_units)

                    units_0 = torch.from_numpy(obs['player_0']['units_mask'][0].astype(np.int8)).view(self.n_units)
                    units_1 = torch.from_numpy(obs['player_1']['units_mask'][1].astype(np.int8)).view(self.n_units)

                    units_position_0 = torch.from_numpy(obs['player_0']['units']['position'][0].astype(np.int8)).view(self.n_units,2)
                    units_position_1 = torch.from_numpy(obs['player_1']['units']['position'][1].astype(np.int8)).view(self.n_units,2)

                    old_units_position_0 = torch.from_numpy(previous_obs['player_0']['units']['position'][0].astype(np.int8)).view(self.n_units,2)
                    old_units_position_1 = torch.from_numpy(previous_obs['player_1']['units']['position'][1].astype(np.int8)).view(self.n_units,2)

                    new_distance_explore_0 = torch.zeros(self.n_units,dtype=torch.int8)
                    new_distance_explore_1 = torch.zeros(self.n_units,dtype=torch.int8)

                    mod_dist_0 = next_state_maps_0[11] + torch.where(next_state_maps_0[9]>0,1,0)*1000
                    special_reward_0 = torch.ones(self.n_units,dtype=torch.int8)
                    mod_dist_1 = next_state_maps_1[11] + torch.where(next_state_maps_1[9]>0,1,0)*1000
                    special_reward_1 = torch.ones(self.n_units,dtype=torch.int8)

                    for unit in torch.argwhere(units_0).view(-1) :
                        for dist in range(0,23) :
                            dx_min = max(units_position_0[unit,0]-dist,0)
                            dx_max = min(units_position_0[unit,0]+dist+1,24)
                            dy_min = max(units_position_0[unit,1]-dist,0)
                            dy_max = min(units_position_0[unit,1]+dist+1,24)
                            dist_tile = mod_dist_0[dx_min:dx_max,dy_min:dy_max]
                            sum_dist = torch.sum(dist_tile)
                            if sum_dist > 0 :
                                new_distance_explore_0[unit] = dist
                                if sum_dist >= 1000 :
                                    special_reward_0[unit] = 10
                                break

                    for unit in torch.argwhere(units_1).view(-1) :
                        for dist in range(0,23) :
                            dx_min = max(units_position_1[unit,0]-dist,0)
                            dx_max = min(units_position_1[unit,0]+dist+1,24)
                            dy_min = max(units_position_1[unit,1]-dist,0)
                            dy_max = min(units_position_1[unit,1]+dist+1,24)
                            dist_tile = mod_dist_1[dx_min:dx_max,dy_min:dy_max]
                            sum_dist = torch.sum(dist_tile)
                            if sum_dist > 0 :
                                new_distance_explore_1[unit] = dist
                                if sum_dist >= 1000 :
                                    special_reward_1[unit] = 10
                                break

                    if non_reset_matchs :

                        #Player 0
                        directions = torch.tensor([[0,0],[0,-1],[1,0],[0,1],[-1,0],[0,0]]).view(6,2)
                        for unit in range(self.n_units) :
                            if action_0[unit,0] == 5 :
                                tx_min = max(units_position_0[unit,0]+action_0[unit,1]-1,0)
                                tx_max = min(units_position_0[unit,0]+action_0[unit,1]+1,23)
                                ty_min = max(units_position_0[unit,1]+action_0[unit,2]-1,0)
                                ty_max = min(units_position_0[unit,1]+action_0[unit,2]+1,23)
                                strike = 0
                                for enemy in range(self.n_units) :
                                    enemy_position =  old_units_position_1[enemy] + directions[action_1[enemy,0]]
                                    if (tx_min<=enemy_position[0]<=tx_max) and (ty_min<=enemy_position[1]<=ty_max) :
                                        bonus = 1
                                        if torch.sum(torch.abs(enemy_position - units_position_1[enemy])) != 0 :
                                            bonus = 4
                                        if next_state_maps_1[9,enemy_position[0],enemy_position[1]] == 1 :
                                            bonus *= 2
                                        strike += bonus
                                rewards[0,step_cpt,unit] += strike / 100

                        units_vision_min_0 = torch.clamp(units_position_0 - self.ep_params['unit_sensor_range'],0,23)
                        units_vision_max_0 = torch.clamp(units_position_0 + self.ep_params['unit_sensor_range'],0,23)

                        dead_units_0 = torch.where(torch.sum(torch.abs(units_position_0-old_units_position_0),dim=1)>1,1,0) * units_0
                        no_point_reward_0 = torch.ones(self.n_units,dtype=torch.int8)
                        point_position = -torch.ones(self.n_units,2)
                        cpt_point = 0

                        for unit in torch.argwhere(units_0).view(-1) :
                            rx_min = units_vision_min_0[unit,0]
                            rx_max = units_vision_max_0[unit,0]+1
                            ry_min = units_vision_min_0[unit,1]
                            ry_max = units_vision_max_0[unit,1]+1
                            new_tiles_0 = next_state_maps_0[0,rx_min:rx_max,ry_min:ry_max]
                            old_tiles_0 = 1-state_maps_0[3,rx_min:rx_max,ry_min:ry_max]
                            rewards[0,step_cpt,unit] += torch.sum(new_tiles_0*old_tiles_0) / (self.map_width*self.map_height)

                            point_reward = next_state_maps_0[9,units_position_0[unit,0],units_position_0[unit,1]] / 100
                            if (point_reward > 0) and (torch.prod(torch.sum(torch.abs(point_position - units_position_0[unit]),dim=1)) != 0) :
                                point_position[cpt_point] = units_position_0[unit]
                                cpt_point += 1
                                no_point_reward_0[unit] = 0
                                rewards[0,step_cpt,unit] += point_reward

                        reward_condition_0 = no_point_reward_0 * (1-dead_units_0) * units_0
                        rewards[0,step_cpt] -=  reward_condition_0 * special_reward_0 * ((new_distance_explore_0 - distance_explore_0) / (self.map_width*self.map_height))
                        rewards[0,step_cpt] +=  reward_condition_0 * (new_energy_0-energy_0) / (self.max_unit_energy*(torch.log(new_energy_0+1)*4+1))
                        rewards[0,step_cpt] -= dead_units_0 /10

                        #Player 1
                        for unit in range(self.n_units) :
                            if action_1[unit,0] == 5 :
                                tx_min = max(units_position_1[unit,0]+action_1[unit,1]-1,0)
                                tx_max = min(units_position_1[unit,0]+action_1[unit,1]+1,23)
                                ty_min = max(units_position_1[unit,1]+action_1[unit,2]-1,0)
                                ty_max = min(units_position_1[unit,1]+action_1[unit,2]+1,23)
                                strike = 0
                                for enemy in range(self.n_units) :
                                    enemy_position =  old_units_position_0[enemy] + directions[action_0[enemy,0]]
                                    if (tx_min<=enemy_position[0]<=tx_max) and (ty_min<=enemy_position[1]<=ty_max) :
                                        bonus = 1
                                        if torch.sum(torch.abs(enemy_position - units_position_0[enemy])) != 0 :
                                            bonus = 4
                                        if next_state_maps_0[9,enemy_position[0],enemy_position[1]] == 1 :
                                            bonus *= 2
                                        strike += bonus
                                rewards[1,step_cpt,unit] += strike / 100

                        units_vision_min_1 = torch.clamp(units_position_1 - self.ep_params['unit_sensor_range'],0,23)
                        units_vision_max_1 = torch.clamp(units_position_1 + self.ep_params['unit_sensor_range'],0,23)

                        dead_units_1 = torch.where(torch.sum(torch.abs(units_position_1-old_units_position_1),dim=1)>1,1,0) * units_1
                        no_point_reward_1 = torch.ones(self.n_units,dtype=torch.int8)
                        point_position = -torch.ones(self.n_units,2)
                        cpt_point = 0
                        
                        for unit in torch.argwhere(units_1).view(-1) :
                            rx_min = units_vision_min_1[unit,0]
                            rx_max = units_vision_max_1[unit,0]+1
                            ry_min = units_vision_min_1[unit,1]
                            ry_max = units_vision_max_1[unit,1]+1
                            new_tiles_1 = next_state_maps_1[0,rx_min:rx_max,ry_min:ry_max]
                            old_tiles_1 = 1-state_maps_1[3,rx_min:rx_max,ry_min:ry_max]
                            rewards[1,step_cpt,unit] += torch.sum(new_tiles_1*old_tiles_1) / (self.map_width*self.map_height)

                            point_reward = next_state_maps_1[9,units_position_1[unit,0],units_position_1[unit,1]] / 100

                            if (point_reward > 0) and (torch.prod(torch.sum(torch.abs(point_position - units_position_1[unit]),dim=1)) != 0) :
                                point_position[cpt_point] = units_position_1[unit]
                                cpt_point += 1
                                no_point_reward_1[unit] = 0
                                rewards[1,step_cpt,unit] += point_reward

                        reward_condition_1 = no_point_reward_1 * (1-dead_units_1) * units_1
                        rewards[1,step_cpt] -= reward_condition_1 * special_reward_1 * ((new_distance_explore_1 - distance_explore_1) / (self.map_width*self.map_height))
                        rewards[1,step_cpt] += reward_condition_1 * (new_energy_1-energy_1) / (self.max_unit_energy*(torch.log(new_energy_1+1)*4+1))
                        rewards[1,step_cpt] -= dead_units_1 /10
                    
                    energy_0 = new_energy_0
                    energy_1 = new_energy_1
                    distance_explore_0 = new_distance_explore_0
                    distance_explore_1 = new_distance_explore_1
                    
                    cumulated_reward[0] += torch.mean(rewards[0,step_cpt])
                    cumulated_reward[1] += torch.mean(rewards[1,step_cpt])

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

                    log_probs[0,step_cpt] = log_prob_0
                    log_probs[1,step_cpt] = log_prob_1

                    state_maps_0 = next_state_maps_0
                    state_features_0 = next_state_features_0
                    state_maps_1 = next_state_maps_1
                    state_features_1 = next_state_features_1

                    step_cpt += 1
                    previous_obs = obs

                self.reward_queue.put(cumulated_reward)
                self.point_queue.put(cumulated_point)

        print(f"Process {self.worker_id} finished")