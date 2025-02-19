from src.luxai_s3.wrappers import LuxAIS3GymEnv
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.multiprocessing as mp
import numpy as np
import random 
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

class Policy(nn.Module) :

    def __init__(self,player) :

        super(Policy,self).__init__()

        if player == 'player_0' :
            self.player_id = 0
        elif player == 'player_1':
            self.player_id = 1
        else :
            raise Exception("Error in player number") 

        self.n_action = 6
        self.n_units = 16
        self.max_sap_range = 8
        self.n_input_maps = 3
        self.max_relic_nodes = 6
        self.map_height = 24
        self.map_width = 24

        self.actor_size = [512,128]
        self.cnn_channels = [3,3,3]
        self.cnn_kernels = [3,5,9]
        self.cnn_strides = [1,1,1]
        self.critic_size = [512,64]

        self.activation = nn.ReLU()
        self.max_pooling = nn.MaxPool2d(kernel_size=2)

        self.cnn_inputs = nn.Conv2d(self.n_input_maps,
                                    self.cnn_channels[0],
                                    kernel_size=self.cnn_kernels[0],
                                    padding=self.cnn_kernels[0]-1,
                                    stride=self.cnn_strides[0],
                                    dtype=torch.float)
        
        self.cnn_hidden = [nn.Conv2d(self.cnn_channels[i],
                                     self.cnn_channels[i+1],
                                     kernel_size=self.cnn_kernels[i+1],
                                     padding=self.cnn_kernels[i+1]-1,
                                    stride=self.cnn_strides[i+1],
                                    dtype=torch.float) for i in range(len(self.cnn_channels)-1)]
        
        state_maps = torch.zeros(self.n_input_maps,self.map_width,self.map_height)
        state_features = torch.zeros(8*self.n_units + self.max_relic_nodes*3 + 6 + 4)
        
        with torch.no_grad() :
            x = self.cnn_inputs(state_maps)
            x = self.max_pooling(x)
            for layer in self.cnn_hidden :
                x = layer(x)
                x = self.max_pooling(x)
            self.n_input_features = x.flatten(start_dim=0).size(0) + state_features.size(0)
        
        self.inputs_actor = nn.Linear(self.n_input_features,self.actor_size[0],dtype=torch.float)
        self.hidden_actor = [nn.Linear(self.actor_size[i],self.actor_size[i+1],dtype=torch.float) for i in range(len(self.actor_size)-1)]

        self.actor_action = nn.Linear(self.actor_size[-1],self.n_action*self.n_units,dtype=torch.float)
        self.actor_dx = nn.Linear(self.actor_size[-1],(self.max_sap_range*2+1)*self.n_units,dtype=torch.float)
        self.actor_dy = nn.Linear(self.actor_size[-1],(self.max_sap_range*2+1)*self.n_units,dtype=torch.float)

        self.inputs_critic = nn.Linear(self.n_input_features,self.critic_size[0],dtype=torch.float)
        self.hidden_critic = [nn.Linear(self.critic_size[i],self.critic_size[i+1],dtype=torch.float) for i in range(len(self.critic_size)-1)]
        self.outputs_critic = nn.Linear(self.critic_size[-1],1,dtype=torch.float)

    def obs_to_state(self,obs:dict,ep_params:dict) -> torch.Tensor:
        list_state_features = []

        state_maps = torch.zeros(3,24,24,dtype=torch.float)

        state_maps[0] = torch.from_numpy(obs['map_features']['energy'].astype(np.float32))/20 #map_energy
        state_maps[1] = torch.from_numpy(obs['sensor_mask'].astype(np.float32)) #sensor_mask
        state_maps[2] = torch.from_numpy(obs['map_features']['tile_type'].astype(np.float32)) #map_tile_type

        #Units
        list_state_features.append(torch.from_numpy(obs['units']['position'].astype(np.float32)).flatten()/24) #position
        list_state_features.append(torch.from_numpy(obs['units']['energy'].astype(np.float32)).flatten()/400) #energy
        list_state_features.append(torch.from_numpy(obs['units_mask'].astype(np.float32)).flatten()) #unit_mask

        list_state_features.append(torch.from_numpy(obs['relic_nodes'].astype(np.float32)).flatten()/24) #relic_nodes
        list_state_features.append(torch.from_numpy(obs['relic_nodes_mask'].astype(np.float32)).flatten()) #relic_nodes_mask

        #Game
        list_state_features.append(torch.from_numpy(obs['team_points'].astype(np.float32)).flatten()/3000) #team_points
        list_state_features.append(torch.from_numpy(obs['team_wins'].astype(np.float32)).flatten()/5) #team_wins

        list_state_features.append(torch.from_numpy(obs['steps'].astype(np.float32)).flatten()/100) #steps
        list_state_features.append(torch.from_numpy(obs['match_steps'].astype(np.float32)).flatten()/5) #match_steps

        list_state_features.append(torch.FloatTensor([ep_params['unit_move_cost'],ep_params['unit_sap_cost'],ep_params['unit_sap_range'],ep_params['unit_sensor_range']])) #Static information about the episode

        state_features = torch.cat(list_state_features)

        return state_maps , state_features

    def training_forward(self,x_maps,x_features,action,mask_action,mask_dx,mask_dy) :

        x = self.activation(self.cnn_inputs(x_maps))
        x = self.max_pooling(x)
        for layer in self.cnn_hidden :
            x = self.activation(layer(x))
            x = self.max_pooling(x)
        x_input = torch.cat((x.flatten(start_dim=1),x_features),dim=-1)

        x = self.activation(self.inputs_actor(x_input))
        for layer in self.hidden_actor :
            x = self.activation(layer(x))

        actor_action = self.actor_action(x).view(-1,self.n_units,self.n_action) + torch.nan_to_num(mask_action*(-torch.inf))
        actor_dx = self.actor_dx(x).view(-1,self.n_units,self.max_sap_range*2+1) + torch.nan_to_num(mask_dx*(-torch.inf))
        actor_dy = self.actor_dy(x).view(-1,self.n_units,self.max_sap_range*2+1) + torch.nan_to_num(mask_dy*(-torch.inf))

        actor_action = F.log_softmax(actor_action,dim=-1)
        actor_dx = F.log_softmax(actor_dx,dim=-1)
        actor_dy = F.log_softmax(actor_dy,dim=-1)

        x = self.activation(self.inputs_critic(x_input))
        for layer in self.hidden_critic :
            x = self.activation(layer(x))
        value = self.outputs_critic(x)

        # Computing log probabilities for the actions

        batch_size = actor_action.size(0)
        n_units = actor_action.size(1)

        step_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, n_units)
        unit_indices = torch.arange(n_units).unsqueeze(0).expand(batch_size, -1) 

        log_prob = torch.sum(actor_action[step_indices,unit_indices, action[:,:, 0]],axis=1,dtype=torch.float)
        log_prob += torch.sum(actor_dx[step_indices,unit_indices, action[:,:, 1]+self.max_sap_range],axis=1,dtype=torch.float)
        log_prob += torch.sum(actor_dy[step_indices,unit_indices, action[:,:, 2]+self.max_sap_range],axis=1,dtype=torch.float)

        return value,log_prob.view(-1,1)

    def forward(self,x_maps,x_features,obs,ep_params) :

        x = self.activation(self.cnn_inputs(x_maps))
        x = self.max_pooling(x)
        for layer in self.cnn_hidden :
            x = self.activation(layer(x))
            x = self.max_pooling(x)

        x_input = torch.cat((x.flatten(start_dim=0),x_features),dim=-1)

        x = self.activation(self.inputs_actor(x_input))
        for layer in self.hidden_actor :
            x = self.activation(layer(x))

        state = {}
        state['energy'] = torch.from_numpy(obs['units']['energy'][self.player_id].astype(int)).view(self.n_units)
        state['units'] = torch.from_numpy(obs['units']['position'][self.player_id].astype(int)).view(self.n_units,2)
        state['map'] = torch.from_numpy(obs['map_features']['tile_type'].astype(int)).view(24,24)
        
        energy_mask = state['energy'] < ep_params['unit_move_cost']
        sap_mask = state['energy'] < ep_params['unit_sap_cost']

        mask_action = torch.zeros(self.n_units,self.n_action,dtype=torch.int8)
        mask_dx = torch.zeros(self.n_units,self.max_sap_range*2+1,dtype=torch.int8)
        mask_dy = torch.zeros(self.n_units,self.max_sap_range*2+1,dtype=torch.int8)

        mask_action[torch.where(energy_mask)[0],1:] += 1
        mask_action[torch.where(sap_mask)[0],-1] += 1

        directions = torch.tensor([[0,-1],[1,0],[0,1],[-1,0]]).view(4,2)
        target_tiles = state['units'].unsqueeze(1).expand(self.n_units, 4, 2) + directions
        clamp_target_tiles = torch.clamp(target_tiles,0,23).view(self.n_units*4,2)
        target_tiles_type = state['map'][clamp_target_tiles[:,0],clamp_target_tiles[:,1]].view(self.n_units,4)

        correct_move_direction = (((target_tiles >= 0) & (target_tiles <= 23)).all(dim=-1)) & (target_tiles_type != 2)
        forbidden_move = 1 - correct_move_direction.int() 
        mask_action[:,1:-1] += forbidden_move
        
        
        actor_action = self.actor_action(x).view(self.n_units,self.n_action) + torch.nan_to_num(mask_action*(-torch.inf))
        actor_action = F.log_softmax(actor_action,dim=-1)
        action_choice = Categorical(logits=actor_action).sample()

        sap_mask =  sap_mask | (action_choice !=5)

        mask_dx[:,:self.max_sap_range-ep_params['unit_sap_range']] += 1
        mask_dx[:,self.max_sap_range+ep_params['unit_sap_range']:] += 1
        mask_dy[:,:self.max_sap_range-ep_params['unit_sap_range']] += 1
        mask_dy[:,self.max_sap_range+ep_params['unit_sap_range']:] += 1

        mask_dx[torch.where(sap_mask)[0]] += 1
        mask_dy[torch.where(sap_mask)[0]] += 1

        directions = torch.arange(-ep_params['unit_sap_range'],ep_params['unit_sap_range']+1).view(2*ep_params['unit_sap_range']+1)
        expand_postion = state['units'].unsqueeze(1).expand(self.n_units,2*ep_params['unit_sap_range']+1,2)
        target_dx = expand_postion[:,:,0].view(self.n_units,2*ep_params['unit_sap_range']+1) + directions
        target_dy = expand_postion[:,:,1].view(self.n_units,2*ep_params['unit_sap_range']+1) + directions

        forbidden_dx = ((target_dx<0) | (target_dx>23)).int()
        forbidden_dy = ((target_dy<0) | (target_dy>23)).int()

        mask_dx[:,self.max_sap_range-ep_params['unit_sap_range']:self.max_sap_range+ep_params['unit_sap_range']+1] += forbidden_dx
        mask_dy[:,self.max_sap_range-ep_params['unit_sap_range']:self.max_sap_range+ep_params['unit_sap_range']+1] += forbidden_dy

        mask_dx[:,self.max_sap_range] = 0
        mask_dy[:,self.max_sap_range] = 0

        actor_dx = self.actor_dx(x).view(self.n_units,self.max_sap_range*2+1) + torch.nan_to_num(mask_dx*(-torch.inf))
        actor_dy = self.actor_dy(x).view(self.n_units,self.max_sap_range*2+1) + torch.nan_to_num(mask_dy*(-torch.inf))
        actor_dx = F.log_softmax(actor_dx,dim=-1)
        actor_dy = F.log_softmax(actor_dy,dim=-1)

        # Sampling action based on the policy
        action = torch.zeros(self.n_units, 3, dtype=torch.int)

        action[:, 0] = action_choice
        action[:, 1] = Categorical(logits=actor_dx).sample() - self.max_sap_range
        action[:, 2] = Categorical(logits=actor_dy).sample() - self.max_sap_range

        x = self.activation(self.inputs_critic(x_input))
        for layer in self.hidden_critic :
            x = self.activation(layer(x))
        value = self.outputs_critic(x)

        return action, value, mask_action, mask_dx, mask_dy
    
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


class ReplayBuffer(Dataset):
    def __init__(self,states_maps,states_features,actions,advantages,returns,mask_action,mask_dx,mask_dy):

        self.states_maps = states_maps
        self.states_features = states_features
        self.actions = actions
        self.advantages = advantages
        self.returns = returns
        self.mask_action = mask_action
        self.mask_dx = mask_dx
        self.mask_dy = mask_dy

    def __len__(self):
        return len(self.states_maps)

    def __getitem__(self, idx):
        return self.states_maps[idx],self.states_features[idx],self.actions[idx],self.advantages[idx],self.returns[idx],self.mask_action[idx],self.mask_dx[idx],self.mask_dy[idx]
    
if __name__ == "__main__":

    print('Initialise training environment...\n')
    lr0 = 1e-6
    lr1 = 1e-7
    max_norm0 = 0.5
    max_norm1 = 0.5
    entropy_coef0 = 0.1
    entropy_coef1 = 0.05

    batch_size = 100
    vf_coef = 0.5
    gamma = 0.99
    gae_lambda = 0.95
    save_rate = 100

    n_epochs = int(1e6)
    n_batch = 10
    num_workers = 6
    n_episode = 4
    n_steps = 100

    file_name = 'experiment_1'
    save_dir = f"policy/{file_name}"
     
    env = LuxAIS3GymEnv(numpy_output=True)
    n_units = env.env_params.max_units
    sap_range = env.env_params.unit_sap_range
    match_step = env.env_params.max_steps_in_match + 1
    len_episode = match_step * env.env_params.match_count_per_episode
    n_action = 6
    n_input = 1880
    victory_bonus = 0
    step_cpt = 0
    reward_cpt = 0
    writer = SummaryWriter(f"runs/{file_name}")

    model_0 = Policy('player_0')
    model_0.share_memory()  # For multiprocessing, the model parameters must be shared
    optimizer_0 = torch.optim.Adam(model_0.parameters(), lr=lr0)

    model_1 = Policy('player_1')
    model_1.share_memory()  # For multiprocessing, the model parameters must be shared
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=lr1)

    shared_queue = mp.Queue()  # Queue to share data between workers and the main process
    reward_queue = mp.Queue()
    event = mp.Event()

    print('Instantiate workers...')
    workers = []
    for i in range(num_workers) :
        worker = Luxai_Worker(i, shared_queue, model_0, model_1, victory_bonus,gamma,gae_lambda,n_steps,reward_queue,event)
        workers.append(worker)
        worker.start()
        print(f'--------worker {i} is ready')
    print('Done\n')
    
    # Main training loop: Collect experiences from workers and update the model
    print('Start training...\n')
    for epoch in range(n_epochs):

        # Collect data from workers
        experiences_0 = []
        experiences_1 = []

        for _ in range(n_episode):
            for _ in range(num_workers) :

                states_maps,states_features,actions,advantages,returns,mask_actions,mask_dxs,mask_dys = shared_queue.get()

                experiences_0.append((states_maps[0],states_features[0],actions[0],advantages[0],returns[0],mask_actions[0],mask_dxs[0],mask_dys[0]))
                experiences_1.append((states_maps[1],states_features[1],actions[1],advantages[1],returns[1],mask_actions[1],mask_dxs[1],mask_dys[1]))

            event.set()

        # Process the collected experiences
        states_maps_0,states_features_0,actions_0,advantages_0,returns_0,mask_action_0,mask_dx_0,mask_dy_0 = zip(*experiences_0)
        states_maps_1,states_features_1,actions_1,advantages_1,returns_1,mask_action_1,mask_dx_1,mask_dy_1 = zip(*experiences_1)

        states_maps_0 = torch.cat(states_maps_0,dim=0)
        states_features_0 = torch.cat(states_features_0,dim=0)
        actions_0 = torch.cat(actions_0,dim=0)
        advantages_0 = torch.cat(advantages_0,dim=0)
        returns_0 = torch.cat(returns_0,dim=0)
        mask_action_0 = torch.cat(mask_action_0,dim=0)
        mask_dx_0 = torch.cat(mask_dx_0,dim=0)
        mask_dy_0 = torch.cat(mask_dy_0,dim=0)

        states_maps_1 = torch.cat(states_maps_1,dim=0)
        states_features_1 = torch.cat(states_features_1,dim=0)
        actions_1 = torch.cat(actions_1,dim=0)
        advantages_1 = torch.cat(advantages_1,dim=0)
        returns_1 = torch.cat(returns_1,dim=0)
        mask_action_1 = torch.cat(mask_action_1,dim=0)
        mask_dx_1 = torch.cat(mask_dx_1,dim=0)
        mask_dy_1 = torch.cat(mask_dy_1,dim=0)

        train_data_0 = ReplayBuffer(states_maps_0,states_features_0,actions_0,advantages_0,returns_0,mask_action_0,mask_dx_0,mask_dy_0)
        train_data_1 = ReplayBuffer(states_maps_1,states_features_1,actions_1,advantages_1,returns_1,mask_action_1,mask_dx_1,mask_dy_1)
 
        train_loader_0 = DataLoader(train_data_0, batch_size=batch_size, shuffle=True)
        train_loader_1 = DataLoader(train_data_1, batch_size=batch_size, shuffle=True)

        for _ in range(n_batch) :

            for batch in train_loader_0 :

                states_maps_,states_features_,actions_,advantages_,returns_,mask_action_,mask_dx_,mask_dy_ = batch

                #Compute log_probs and values
                values_,log_probs_ = model_0.training_forward(states_maps_,states_features_,actions_,mask_action_,mask_dx_,mask_dy_)

                advantages_ = (advantages_ - advantages_.mean()) / (advantages_.std() + 1e-8)

                # Losses
                entropy_loss_0 = -torch.mean(torch.exp(log_probs_) * log_probs_)
                policy_loss_0 = -torch.mean(log_probs_ * advantages_)
                value_loss_0 = F.mse_loss(values_, returns_)

                loss_0 = policy_loss_0 + vf_coef *value_loss_0 + entropy_coef0 * entropy_loss_0

                # Update model
                optimizer_0.zero_grad()
                loss_0.backward()
                torch.nn.utils.clip_grad_norm_(model_0.parameters(), max_norm=max_norm0)
                optimizer_0.step()

            for batch in train_loader_1 :

                states_maps_,states_features_,actions_,advantages_,returns_,mask_action_,mask_dx_,mask_dy_ = batch

                #Compute log_probs and values
                values_,log_probs_ = model_1.training_forward(states_maps_,states_features_,actions_,mask_action_,mask_dx_,mask_dy_)

                advantages_ = (advantages_ - advantages_.mean()) / (advantages_.std() + 1e-8)

                # Losses
                entropy_loss_1 = -torch.mean(torch.exp(log_probs_) * log_probs_)
                policy_loss_1 = -torch.mean(log_probs_ * advantages_)
                value_loss_1 = F.mse_loss(values_, returns_)

                loss_1 = policy_loss_1 + vf_coef *value_loss_1 + entropy_coef1 * entropy_loss_1

                # Update model
                optimizer_1.zero_grad()
                loss_1.backward()
                torch.nn.utils.clip_grad_norm_(model_1.parameters(), max_norm=max_norm1)
                optimizer_1.step()

        try :
            while True :
                reward = reward_queue.get_nowait()
                reward_cpt += len_episode
                writer.add_scalar("Reward 0", reward[0].item(), reward_cpt)
                writer.add_scalar("Reward 1", reward[1].item(), reward_cpt)
        except :
            None

        step_cpt += n_episode*n_steps*num_workers

        writer.add_scalar("Loss/Total Loss 0", loss_0.item(), step_cpt)
        writer.add_scalar("Loss/Total Loss 1", loss_1.item(), step_cpt)

        writer.add_scalar("Loss/Policy Loss 0", policy_loss_0.item(), step_cpt)
        writer.add_scalar("Loss/Policy Loss 1", policy_loss_1.item(), step_cpt)

        writer.add_scalar("Loss/Value Loss 0", value_loss_0.item(), step_cpt)
        writer.add_scalar("Loss/Value Loss 1", value_loss_1.item(), step_cpt)

        writer.add_scalar("Loss/Entropy Loss 0", entropy_loss_0.item(), step_cpt)
        writer.add_scalar("Loss/Entropy Loss 1", entropy_loss_1.item(), step_cpt)

        print(f"Episode {epoch}, Loss 0 : {loss_0.item()}, Loss 1 : {loss_1.item()}")

        del train_data_1, train_data_0, train_loader_0, train_loader_1

        if epoch % save_rate == 0 :
            if not os.path.exists(save_dir):
                # Créer le dossier
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, f"policy_0_epoch_{epoch}.pth")
            torch.save(model_0.state_dict(), save_path)

            save_path = os.path.join(save_dir, f"policy_1_epoch_{epoch}.pth")
            torch.save(model_1.state_dict(), save_path)

    # Terminate workers
    for worker in workers :
        worker.join()
    
    writer.close()