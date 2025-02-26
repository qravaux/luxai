import scipy.signal
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.distributions import Categorical
import scipy
import matplotlib.pyplot as plt

class Luxai_Agent(nn.Module) :

    def __init__(self,player) :

        super(Luxai_Agent,self).__init__()

        if player == 'player_0' :
            self.player_id = 0
        elif player == 'player_1':
            self.player_id = 1
        else :
            raise Exception("Error in player number") 

        self.n_action = 6
        self.n_units = 16
        self.max_sap_range = 8
        self.n_input_maps = 12
        self.max_relic_nodes = 6
        self.map_height = 24
        self.map_width = 24

        self.actor_size = [2048,512,128]
        self.cnn_channels = [8,16,32]
        self.cnn_kernels = [9,5,3]
        self.cnn_strides = [1,1,1]
        self.critic_size = [2048,512,128]

        self.activation = nn.Tanh()
        self.gain = 5/3 #For Tanh and sqrt(2) for ReLu

        self.max_pooling = nn.MaxPool2d(kernel_size=2)

        self.cnn_inputs = nn.Conv2d(self.n_input_maps,
                                    self.cnn_channels[0],
                                    kernel_size=self.cnn_kernels[0],
                                    padding=self.cnn_kernels[0]-1,
                                    stride=self.cnn_strides[0],
                                    dtype=torch.float)
        
        self.cnn_hidden = nn.ModuleList([nn.Conv2d(self.cnn_channels[i],
                                     self.cnn_channels[i+1],
                                     kernel_size=self.cnn_kernels[i+1],
                                     padding=self.cnn_kernels[i+1]-1,
                                    stride=self.cnn_strides[i+1],
                                    dtype=torch.float) for i in range(len(self.cnn_channels)-1)])
        
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
        self.hidden_actor = nn.ModuleList([nn.Linear(self.actor_size[i],self.actor_size[i+1],dtype=torch.float) for i in range(len(self.actor_size)-1)])

        self.actor_action = nn.Linear(self.actor_size[-1],self.n_action*self.n_units,dtype=torch.float)
        self.actor_dx = nn.Linear(self.actor_size[-1],(self.max_sap_range*2+1)*self.n_units,dtype=torch.float)
        self.actor_dy = nn.Linear(self.actor_size[-1],(self.max_sap_range*2+1)*self.n_units,dtype=torch.float)

        self.inputs_critic = nn.Linear(self.n_input_features,self.critic_size[0],dtype=torch.float)
        self.hidden_critic = nn.ModuleList([nn.Linear(self.critic_size[i],self.critic_size[i+1],dtype=torch.float) for i in range(len(self.critic_size)-1)])
        self.outputs_critic = nn.Linear(self.critic_size[-1],self.n_units,dtype=torch.float)

        #Initialization
        nn.init.orthogonal_(self.cnn_inputs.weight,gain=self.gain)
        nn.init.zeros_(self.cnn_inputs.bias)

        for layers in self.cnn_hidden :
            nn.init.orthogonal_(layers.weight,gain=self.gain)
            nn.init.zeros_(layers.bias)

        nn.init.orthogonal_(self.inputs_actor.weight,gain=self.gain)
        nn.init.zeros_(self.inputs_actor.bias)

        for layers in self.hidden_actor :
            nn.init.orthogonal_(layers.weight,gain=self.gain)
            nn.init.zeros_(layers.bias)

        nn.init.orthogonal_(self.inputs_critic.weight,gain=self.gain)
        nn.init.zeros_(self.inputs_critic.bias)

        for layers in self.hidden_critic :
            nn.init.orthogonal_(layers.weight,gain=self.gain)
            nn.init.zeros_(layers.bias)

        nn.init.orthogonal_(self.actor_action.weight,gain=0.01)
        nn.init.zeros_(self.actor_action.bias)

        nn.init.orthogonal_(self.actor_dx.weight,gain=0.01)
        nn.init.zeros_(self.actor_dx.bias)

        nn.init.orthogonal_(self.actor_dy.weight,gain=0.01)
        nn.init.zeros_(self.actor_dy.bias)

        nn.init.orthogonal_(self.outputs_critic.weight,gain=1)
        nn.init.zeros_(self.outputs_critic.bias)

    def obs_to_state(self,obs:dict,ep_params:dict,points=0,map_memory=None,show=False) -> torch.Tensor:

        if map_memory == None :
            map_memory = torch.zeros(12,24,24,dtype=torch.float)
            map_memory[5] = -1
            map_memory[4] = -1
            map_memory[9] = -1

        list_state_features = []

        state_maps = torch.zeros(12,24,24,dtype=torch.float)

        state_maps[0] = torch.from_numpy(obs['sensor_mask'].astype(np.float32)) #sensor_mask
        state_maps[1] = torch.from_numpy(obs['map_features']['energy'].astype(np.float32))/20 #map_energy
        state_maps[2] = torch.from_numpy(obs['map_features']['tile_type'].astype(np.float32))/2 #map_tile_type

        for (i,j) in obs['units']['position'][self.player_id] :
            if i != -1 :
                state_maps[7,i,j] = 1
        
        #Compute memory map
        state_maps[3] = state_maps[0] + (1-state_maps[0]) * torch.rot90(state_maps[0],2).T #Because the map is symetric
        state_maps[4] = state_maps[1] * state_maps[0] + (1-state_maps[0]) * torch.rot90(state_maps[1],2).T
        state_maps[5] = state_maps[2] * state_maps[0] + (1-state_maps[0]) * torch.rot90(state_maps[2],2).T

        state_maps[4] = state_maps[3] * state_maps[4] + (1-state_maps[3]) * map_memory[4]
        state_maps[5] = state_maps[3] * state_maps[5] + (1-state_maps[3]) * map_memory[5]
        state_maps[3] = state_maps[3] + (1-state_maps[3]) * map_memory[3] #Add memory
    
        modified_sensor_mask = torch.clamp(torch.from_numpy(scipy.signal.convolve2d(state_maps[7],torch.ones(2*ep_params['unit_sensor_range']+1,2*ep_params['unit_sensor_range']+1),mode="same",boundary="fill",fillvalue=0)),max=1)
        state_maps[10] = modified_sensor_mask + (1-modified_sensor_mask) * torch.rot90(modified_sensor_mask,2).T
        state_maps[10] = state_maps[10] + (1-state_maps[10]) * map_memory[10]
        
        state_maps[6] = map_memory[6]
        for (i,j) in obs['relic_nodes'] :
            if i != -1 :
                state_maps[6,i,j] = 1
                state_maps[6,23-j,23-i] = 1

        if torch.sum(state_maps[6]) == self.max_relic_nodes or torch.sum(state_maps[3]) == self.map_width*self.map_height :
            map_relic = state_maps[6] 
            map_relic_modified = state_maps[6]
        else :
            map_relic = state_maps[6] + (1-state_maps[3])
            map_relic_modified = state_maps[6] + (1-state_maps[10])

        state_maps[8] = torch.clamp(torch.from_numpy(scipy.signal.convolve2d(map_relic,torch.ones(5,5),mode="same",boundary="fill",fillvalue=0)),max=1)
        state_maps[8] = state_maps[8] * torch.where(map_memory[9]!=0,1,0)

        state_maps[11] = torch.clamp(torch.from_numpy(scipy.signal.convolve2d(map_relic_modified,torch.ones(5,5),mode="same",boundary="fill",fillvalue=0)),max=1)
        state_maps[11] = state_maps[11] * torch.where(map_memory[9]!=0,1,0) * (1-state_maps[7])

        question = state_maps[8] * state_maps[7]
        old_prob = map_memory[9] * state_maps[8]
        new_prob = question*(points/torch.sum(question)) + (1-question)*old_prob
        state_maps[9] = torch.where(new_prob > old_prob, new_prob, old_prob)
        state_maps[9] = torch.where(new_prob==0,new_prob,state_maps[9])
        rotated_map = torch.rot90(state_maps[9],2).T
        state_maps[9] = torch.where(state_maps[9] > rotated_map, state_maps[9], rotated_map)

        #Units
        list_state_features.append(torch.from_numpy(obs['units']['position'].astype(np.float32)).flatten()/24) #position
        list_state_features.append(torch.from_numpy(obs['units']['energy'].astype(np.float32)).flatten()/400) #energy
        list_state_features.append(torch.from_numpy(obs['units_mask'].astype(np.float32)).flatten()) #unit_mask

        list_state_features.append(torch.from_numpy(obs['relic_nodes'].astype(np.float32)).flatten()/24) #relic_nodes
        list_state_features.append(torch.from_numpy(obs['relic_nodes_mask'].astype(np.float32)).flatten()) #relic_nodes_mask

        #Game
        list_state_features.append(torch.from_numpy(obs['team_points'].astype(np.float32)).flatten()/3000) #team_points
        list_state_features.append(torch.from_numpy(obs['team_wins'].astype(np.float32)).flatten()/5) #team_wins

        list_state_features.append(torch.from_numpy(obs['steps'].astype(np.float32)).flatten()/505) #steps
        list_state_features.append(torch.from_numpy(obs['match_steps'].astype(np.float32)).flatten()/101) #match_steps

        list_state_features.append(torch.FloatTensor([ep_params['unit_move_cost'],ep_params['unit_sap_cost'],ep_params['unit_sap_range'],ep_params['unit_sensor_range']])) #Static information about the episode

        state_features = torch.cat(list_state_features)

        if show :     
            plt.imshow(state_maps[0].T)
            plt.show()
            plt.imshow(modified_sensor_mask.T)
            plt.show()
            for i in range(1,12) :
                plt.imshow(state_maps[i].T)
                plt.show()

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

        actor_action = torch.nan_to_num(self.actor_action(x).view(self.n_units,self.n_action) - torch.nan_to_num(mask_action*torch.inf))
        actor_dx = torch.nan_to_num(self.actor_dx(x).view(self.n_units,self.max_sap_range*2+1) - torch.nan_to_num(mask_dx*torch.inf))
        actor_dy = torch.nan_to_num(self.actor_dy(x).view(self.n_units,self.max_sap_range*2+1) - torch.nan_to_num(mask_dy*torch.inf))

        actor_action = torch.nan_to_num(F.log_softmax(actor_action,dim=-1))
        actor_dx = torch.nan_to_num(F.log_softmax(actor_dx,dim=-1))
        actor_dy = torch.nan_to_num(F.log_softmax(actor_dy,dim=-1))

        x = self.activation(self.inputs_critic(x_input))
        for layer in self.hidden_critic :
            x = self.activation(layer(x))
        value = self.outputs_critic(x)

        # Computing log probabilities for the actions

        batch_size = actor_action.size(0)

        step_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, self.n_units)
        unit_indices = torch.arange(self.n_units).unsqueeze(0).expand(batch_size, -1) 

        log_prob = actor_action[step_indices,unit_indices, action[:,:, 0]]
        log_prob += actor_dx[step_indices,unit_indices, action[:,:, 1]+self.max_sap_range]
        log_prob += actor_dy[step_indices,unit_indices, action[:,:, 2]+self.max_sap_range]

        return value,log_prob

    def forward(self,x_maps,x_features,obs,ep_params) :

        with torch.no_grad() :

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
            
            """
            mask_action = torch.ones(self.n_units,self.n_action,dtype=torch.int8)
            mask_action[:,0] = 0
            mask_dx = torch.ones(self.n_units,self.max_sap_range*2+1,dtype=torch.int8)
            mask_dy = torch.ones(self.n_units,self.max_sap_range*2+1,dtype=torch.int8)

            mask_action[0] = 0
            mask_dx[0] = 0
            mask_dy[0] = 0
            """

            mask_action[torch.argwhere(energy_mask),1:] += 1
            mask_action[torch.argwhere(sap_mask),-1] += 1

            directions = torch.tensor([[0,-1],[1,0],[0,1],[-1,0]]).view(4,2)
            target_tiles = (state['units'].unsqueeze(1).expand(self.n_units, 4, 2) + directions).clone()
            clamp_target_tiles = torch.clamp(target_tiles,0,23).view(self.n_units*4,2)
            target_tiles_type = state['map'][clamp_target_tiles[:,0],clamp_target_tiles[:,1]].view(self.n_units,4)

            correct_move_direction = (((target_tiles >= 0) & (target_tiles <= 23)).all(dim=-1)) & (target_tiles_type != 2)
            forbidden_move = 1 - correct_move_direction.int() 
            mask_action[:,1:-1] += forbidden_move
            
            
            actor_action = torch.nan_to_num(self.actor_action(x).view(self.n_units,self.n_action) - torch.nan_to_num(mask_action*torch.inf))
            actor_action = torch.nan_to_num(F.log_softmax(actor_action,dim=-1))
            action_choice = Categorical(logits=actor_action).sample()

            sap_mask =  sap_mask | (action_choice !=5)

            mask_dx[:,:self.max_sap_range-ep_params['unit_sap_range']] += 1
            mask_dx[:,self.max_sap_range+ep_params['unit_sap_range']:] += 1
            mask_dy[:,:self.max_sap_range-ep_params['unit_sap_range']] += 1
            mask_dy[:,self.max_sap_range+ep_params['unit_sap_range']:] += 1

            mask_dx[torch.where(sap_mask)[0]] += 1
            mask_dy[torch.where(sap_mask)[0]] += 1

            directions = torch.arange(-ep_params['unit_sap_range'],ep_params['unit_sap_range']+1).view(2*ep_params['unit_sap_range']+1)
            expand_postion = state['units'].unsqueeze(1).expand(self.n_units,2*ep_params['unit_sap_range']+1,2).clone()
            target_dx = expand_postion[:,:,0].view(self.n_units,2*ep_params['unit_sap_range']+1) + directions
            target_dy = expand_postion[:,:,1].view(self.n_units,2*ep_params['unit_sap_range']+1) + directions

            forbidden_dx = ((target_dx<0) | (target_dx>23)).int()
            forbidden_dy = ((target_dy<0) | (target_dy>23)).int()

            mask_dx[:,self.max_sap_range-ep_params['unit_sap_range']:self.max_sap_range+ep_params['unit_sap_range']+1] += forbidden_dx
            mask_dy[:,self.max_sap_range-ep_params['unit_sap_range']:self.max_sap_range+ep_params['unit_sap_range']+1] += forbidden_dy

            mask_dx[:,self.max_sap_range] = 0
            mask_dy[:,self.max_sap_range] = 0

            actor_dx = torch.nan_to_num(self.actor_dx(x).view(self.n_units,self.max_sap_range*2+1) - torch.nan_to_num(mask_dx*torch.inf))
            actor_dy = torch.nan_to_num(self.actor_dy(x).view(self.n_units,self.max_sap_range*2+1) - torch.nan_to_num(mask_dy*torch.inf))
            actor_dx = torch.nan_to_num(F.log_softmax(actor_dx,dim=-1))
            actor_dy = torch.nan_to_num(F.log_softmax(actor_dy,dim=-1))

            # Sampling action based on the policy
            action = torch.zeros(self.n_units, 3, dtype=torch.int)

            action[:, 0] = action_choice
            action[:, 1] = Categorical(logits=actor_dx).sample() - self.max_sap_range
            action[:, 2] = Categorical(logits=actor_dy).sample() - self.max_sap_range

            x = self.activation(self.inputs_critic(x_input))
            for layer in self.hidden_critic :
                x = self.activation(layer(x))
            value = self.outputs_critic(x)

            # Computing log probabilities for the actions

            unit_indices = torch.arange(self.n_units) 

            log_prob = actor_action[unit_indices, action[:, 0]]
            log_prob += actor_dx[unit_indices, action[:, 1]+self.max_sap_range]
            log_prob += actor_dy[unit_indices, action[:, 2]+self.max_sap_range]

        return action, value, mask_action, mask_dx, mask_dy, log_prob
    