import torch
import jax
import jax.numpy as jnp
import jax.nn as nn
from src.luxai_s3.params import EnvParams, env_params_ranges

class PrioritizedReplayBuffer:
    def __init__(self, capacity,device, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.position = 0
        self.len_cpt = 0
        self.priorities = torch.zeros(capacity, dtype=torch.float32).to(device)
        self.priorities[0] = 1
        self.n_inputs_maps = 10
        self.n_units = 16
        self.n_action = 6
        self.sap_range = 8
        self.n_inputs_features = 156
        self.map_width = 24
        self.map_height = 24

        self.states_maps_buffer = torch.zeros(self.capacity,self.n_inputs_maps,self.map_width,self.map_height,dtype=torch.float32).to(device)
        self.states_features_buffer = torch.zeros(self.capacity,self.n_inputs_features,dtype=torch.float32).to(device) 
        self.actions_buffer = torch.zeros(self.capacity,self.n_units,3,dtype=torch.int32).to(device)
        self.advantages_buffer = torch.zeros(self.capacity,self.n_units,dtype=torch.float32).to(device)
        self.returns_buffer = torch.zeros(self.capacity,self.n_units,dtype=torch.float32).to(device)
        self.log_probs_buffer = torch.zeros(self.capacity,self.n_units,dtype=torch.float32).to(device)
        self. mask_actions_buffer = torch.zeros(self.capacity,self.n_units,self.n_action,dtype=torch.float32).to(device)
        self.mask_dxs_buffer = torch.zeros(self.capacity,self.n_units,self.sap_range*2+1,dtype=torch.float32).to(device)
        self.mask_dys_buffer = torch.zeros(self.capacity,self.n_units,self.sap_range*2+1,dtype=torch.float32).to(device)

        self.priorities.requires_grad = False
        self.advantages_buffer.requires_grad = False
        self.returns_buffer.requires_grad = False
        self.log_probs_buffer.requires_grad = False
        self.states_maps_buffer.requires_grad = False
        self.states_features_buffer.requires_grad = False

    def add(self,states_maps,states_features,actions,advantages,returns,log_probs,mask_actions,mask_dxs,mask_dys):
        max_prio = torch.max(self.priorities)
        n_add = len(states_maps)
        if n_add + self.position <= self.capacity :

            self.states_maps_buffer[self.position:self.position+n_add] = states_maps
            self.states_features_buffer[self.position:self.position+n_add] = states_features
            self.actions_buffer[self.position:self.position+n_add] = actions
            self.advantages_buffer[self.position:self.position+n_add] = advantages
            self.returns_buffer[self.position:self.position+n_add] = returns
            self.log_probs_buffer[self.position:self.position+n_add] = log_probs
            self. mask_actions_buffer[self.position:self.position+n_add] = mask_actions
            self.mask_dxs_buffer[self.position:self.position+n_add] = mask_dxs
            self.mask_dys_buffer[self.position:self.position+n_add] = mask_dys
            self.priorities[self.position:self.position+n_add] = max_prio

        else:
            n_over = n_add + self.position - self.capacity

            self.states_maps_buffer[self.position:] = states_maps[:n_add-n_over]
            self.states_features_buffer[self.position:] = states_features[:n_add-n_over]
            self.actions_buffer[self.position:] = actions[:n_add-n_over]
            self.advantages_buffer[self.position:] = advantages[:n_add-n_over]
            self.returns_buffer[self.position:] = returns[:n_add-n_over]
            self.log_probs_buffer[self.position:] = log_probs[:n_add-n_over]
            self. mask_actions_buffer[self.position:] = mask_actions[:n_add-n_over]
            self.mask_dxs_buffer[self.position:] = mask_dxs[:n_add-n_over]
            self.mask_dys_buffer[self.position:] = mask_dys[:n_add-n_over]
            self.priorities[self.position:] = max_prio

            self.states_maps_buffer[:n_over] = states_maps[n_add-n_over:]
            self.states_features_buffer[:n_over] = states_features[n_add-n_over:]
            self.actions_buffer[:n_over] = actions[n_add-n_over:]
            self.advantages_buffer[:n_over] = advantages[n_add-n_over:]
            self.returns_buffer[:n_over] = returns[n_add-n_over:]
            self.log_probs_buffer[:n_over] = log_probs[n_add-n_over:]
            self. mask_actions_buffer[:n_over] = mask_actions[n_add-n_over:]
            self.mask_dxs_buffer[:n_over] = mask_dxs[n_add-n_over:]
            self.mask_dys_buffer[:n_over] = mask_dys[n_add-n_over:]
            self.priorities[:n_over] = max_prio
        
        self.position = (self.position + n_add) % self.capacity
        if self.len_cpt < self.capacity :
            self.len_cpt = min(n_add + self.len_cpt,self.capacity)

    def sample(self, batch_size, device, beta=0.4):
        if self.len_cpt == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = probs.multinomial(batch_size,replacement=True)

        total = self.len_cpt
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = weights.detach().to(device)

        states_maps = self.states_maps_buffer[indices]
        states_features = self.states_features_buffer[indices]
        actions = self.actions_buffer[indices]
        advantages = self.advantages_buffer[indices]
        returns = self.returns_buffer[indices]
        log_probs  = self.log_probs_buffer[indices]
        mask_actions = self. mask_actions_buffer[indices]
        mask_dxs = self.mask_dxs_buffer[indices]
        mask_dys = self.mask_dys_buffer[indices]

        return states_maps, states_features, actions, advantages, returns, log_probs, mask_actions, mask_dxs, mask_dys, weights, indices

    
    def update_priorities(self, batch_indices, batch_priorities):

        self.priorities[batch_indices] = batch_priorities

def obs_to_state(obs,ep_params,points,map_memory,player):

    obs = obs[player]
    map_width = 24

    list_state_features = []
    state_maps = jnp.zeros((10, map_width, map_width), dtype=jnp.float32)

    if player == "player_0" :
        state_maps = state_maps.at[0].set(jnp.where(obs.sensor_mask,1,0))  # sensor_mask
        state_maps = state_maps.at[1].set(obs.map_features.energy / 20)  # map_energy
        state_maps = state_maps.at[2].set(obs.map_features.tile_type)  # map_tile_type
    else :
        state_maps = state_maps.at[0].set(jnp.flip(jnp.where(obs.sensor_mask,1,0),axis=[-2,-1]).T)  # sensor_mask
        state_maps = state_maps.at[1].set(jnp.flip(obs.map_features.energy / 20,axis=[-2,-1]).T)  # map_energy
        state_maps = state_maps.at[2].set(jnp.flip(obs.map_features.tile_type,axis=[-2,-1]).T)  # map_tile_type

    state_maps = state_maps.at[6].set(map_memory[6])
    
    relic_nodes_mask = jnp.where(obs.relic_nodes_mask,1,0)
    for i, (x,y) in enumerate(obs.relic_nodes) :
        state_maps = state_maps.at[6,x,y].add(relic_nodes_mask[i])
        state_maps = state_maps.at[6, 23-y,23-x].add(relic_nodes_mask[i])
    
    if player == 'player_0' :
        units_mask = jnp.where(obs.units_mask[0],1,0)
        for i, (x,y) in enumerate(obs.units.position[0]) :
            state_maps = state_maps.at[x,y].set(units_mask[i])
    else :
        position_player_1 = obs.units.position[::-1,:,::-1]
        position_player_1 = jnp.where(position_player_1==-1,-1,23-position_player_1)
        units_mask = jnp.where(obs.units_mask[1],1,0)
        for i, (x,y) in enumerate(position_player_1[0]) :
            state_maps = state_maps.at[x,y].set(units_mask[i])
    
    step_match = obs.match_steps
    n_match = jnp.sum(obs.team_wins)
    n_relic = jnp.sum(state_maps[6])

    # Compute memory map
    state_maps = state_maps.at[3].set(state_maps[0] + (1 - state_maps[0]) * jnp.flip(state_maps[0],axis=[-2,-1]).T)  # Because the map is symmetric
    state_maps = state_maps.at[4].set(state_maps[1] + (1 - state_maps[0]) * jnp.flip(state_maps[1],axis=[-2,-1]).T)
    state_maps = state_maps.at[5].set(state_maps[2] + (1 - state_maps[0]) * jnp.flip(state_maps[2],axis=[-2,-1]).T)

    begin_match = jnp.where(step_match==0,1,0)
    map_memory = map_memory.at[3].set(1-begin_match)

    find_all_relics = jnp.where(n_relic==6,1,0) + jnp.where(n_relic==((n_match + 1)*2),1,0)
    map_memory = jnp.clip(map_memory.at[3].add(find_all_relics),max=1)

    state_maps = state_maps.at[4].set(state_maps[4] + (1 - state_maps[3]) * map_memory[4])  # Add memory
    state_maps = state_maps.at[5].set(state_maps[5] + (1 - state_maps[3]) * map_memory[5])
    state_maps = state_maps.at[3].set(map_memory[3] + (1 - map_memory[3]) * state_maps[3])

    reset_point_prob = 1 - (jnp.where(n_match >= 3,1,0)*jnp.where(step_match > 0,1,0) + find_all_relics)

    map_memory = map_memory.at[9].set(jnp.where(map_memory[9] == 0, -2, map_memory[9])*reset_point_prob)
    map_relic = state_maps[6] + (1 - state_maps[0])*reset_point_prob + (1 - state_maps[3])*(1-reset_point_prob)

    state_maps = state_maps.at[8].set(jnp.clip(jax.scipy.signal.convolve2d(map_relic, jnp.ones((5, 5)), mode="same", boundary="fill", fillvalue=0), a_max=1))
    state_maps = state_maps.at[8].set(state_maps[8] * jnp.where(map_memory[9] != 0, 1, 0))

    question = state_maps[8] * state_maps[7]
    old_prob = map_memory[9] * state_maps[8]
    sure = jnp.sum(jnp.where(map_memory[9] * question == 1, 1, 0))
    sum_question = jnp.sum(question)

    points_bool = jnp.clip(jnp.where(points==0,1,0) + jnp.where(sum_question-sure==0,1,0))
    points_prob = (1-points_bool) * ((points - sure) / (sum_question-sure+points_bool))

    new_prob = question * points_prob + (1 - question) * (-1)

    state_maps = state_maps.at[9].set(jnp.where(new_prob > old_prob, new_prob, old_prob))
    rotated_map = jnp.flip(state_maps[9],axis=[-2,-1]).T
    state_maps = state_maps.at[9].set(jnp.where(state_maps[9] >= rotated_map, state_maps[9], rotated_map))

    
    if player == 'player_0' :
        # Units
        list_state_features.append(obs.units.position.flatten() / map_width)  # position
        list_state_features.append(obs.units.energy.flatten() / 400)  # energy
        list_state_features.append(jnp.where(obs.units_mask,1,0).flatten())  # unit_mask

        # Game
        list_state_features.append(obs.team_points.flatten() / 3000)  # team_points
        list_state_features.append(obs.team_wins.flatten() / 5)  # team_wins

    else :
        # Units
        list_state_features.append(position_player_1.flatten() / map_width)  # position
        list_state_features.append(obs.units.energy[::-1].flatten() / 400)  # energy
        list_state_features.append(jnp.where(obs.units_mask,1,0)[::-1].flatten())  # unit_mask

        # Game
        list_state_features.append(obs.team_points[::-1].flatten() / 3000)  # team_points
        list_state_features.append(obs.team_wins[::-1].flatten() / 5)  # team_wins

    list_state_features.append(obs.relic_nodes.flatten() / map_width)  # relic_nodes
    list_state_features.append(jnp.where(obs.relic_nodes_mask,1,0).flatten())  # relic_nodes_mask 

    list_state_features.append(obs.steps.flatten() / 505)  # steps
    list_state_features.append(obs.match_steps.flatten() / 101)  # match_steps

    list_state_features.append(jnp.array([ep_params.unit_move_cost/6, 
                                          ep_params.unit_sap_cost/51, 
                                          ep_params.unit_sap_range/8, 
                                          ep_params.unit_sensor_range/4], dtype=jnp.float32))  # Static information about the episode

    state_features = jnp.concatenate(list_state_features,dtype=jnp.float32)

    return state_maps, state_features

def random_params(rng_key) : 
    randomized_game_params = dict()
    for k, v in env_params_ranges.items():
        rng_key, subkey = jax.random.split(rng_key)
        randomized_game_params[k] = jax.random.choice(
            subkey, jax.numpy.array(v)
        )
    params = EnvParams(**randomized_game_params)
    return params

def generate_map_memory(num_envs) :
    map_memory = jnp.zeros((num_envs, 10, 24, 24), dtype=jnp.float32)
    map_memory = map_memory.at[:,5].set(-1)
    map_memory = map_memory.at[:,4].set(-1)
    map_memory = map_memory.at[:,9].set(-1)

    points = jnp.zeros((num_envs,1), dtype=jnp.float32)

    return map_memory, points

def generate_random_action(rng_key) :
    action = dict(
                player_0=jax.random.randint(rng_key, (16, 3), 0, 4),
                player_1=jax.random.randint(rng_key, (16, 3), 0, 4)
            )
    return action

def compute_mask_actions_log_probs(obs,ep_params,action_key,actor_action,actor_dx,actor_dy,player) :
    obs = obs[player]
    player_id = jnp.where(player=='player_1',1,0)
    n_units = 16
    n_action = 6
    max_sap_range = 8

    if player=='player_0' :
        position = obs.units.position[player_id]
    else :
        position = obs.units.position[player_id,:,::-1]
        position = jnp.where(position==-1,-1,23-position)
    
    #Gather information for masking
    energy_mask = jnp.expand_dims(jnp.where(obs.units.energy[player_id]<ep_params.unit_move_cost,1,0),axis=-1)
    sap_mask = jnp.where(obs.units.energy[player_id]<ep_params.unit_sap_cost,1,0)

    #Compute action masks
    mask_action = jnp.zeros((n_units,n_action),dtype=jnp.int32)
    mask_dx = jnp.zeros((n_units,max_sap_range*2+1),dtype=jnp.int32)
    mask_dy = jnp.zeros((n_units,max_sap_range*2+1),dtype=jnp.int32)

    mask_action = mask_action.at[:,1:].add(energy_mask)
    mask_action = mask_action.at[:,-1].add(sap_mask)

    directions = jnp.array([[0,-1],[1,0],[0,1],[-1,0]])

    target_tiles = jnp.repeat(jnp.expand_dims(position,axis=-2),4,axis=-2) + directions

    clamp_target_tiles = jnp.clip(target_tiles,min=0,max=23).reshape(n_units*4,2)
    target_tiles_type = obs.map_features.tile_type[clamp_target_tiles[:,0],clamp_target_tiles[:,1]].reshape(n_units,4)

    out_board_tiles = jnp.where(target_tiles<0,1,0) + jnp.where(target_tiles>23,1,0)
    forbidden_move = out_board_tiles[:,:,0] + out_board_tiles[:,:,1] + jnp.where(target_tiles_type==2,1,0)
    mask_action = mask_action.at[:,1:-1].add(forbidden_move)

    actor_action = actor_action - mask_action*100
    actor_action = nn.log_softmax(actor_action,axis=-1)
    action_key, action_choice_key = jax.random.split(action_key)
    action_choice = jax.random.categorical(key=action_choice_key, logits=actor_action, axis=-1)

    sap_mask =  jnp.expand_dims(sap_mask+jnp.where(action_choice !=5,1,0),axis=-1)
    
    sap_range = jnp.arange(-max_sap_range,max_sap_range+1)
    forbidden_range = jnp.where(sap_range<-ep_params.unit_sap_range,1,0) + jnp.where(ep_params.unit_sap_range<sap_range,1,0)

    mask_dx = mask_dx.at[:].add(forbidden_range) 
    mask_dy = mask_dy.at[:].add(forbidden_range)

    mask_dx = mask_dx.at[:].add(sap_mask) 
    mask_dy = mask_dy.at[:].add(sap_mask)

    expand_postion = jnp.repeat(jnp.expand_dims(position,axis=-2),2*max_sap_range+1,axis=-2)
    target_dx = expand_postion[:,:,0] + sap_range
    target_dy = expand_postion[:,:,1] + sap_range

    forbidden_dx = jnp.where(target_dx<0,1,0) + jnp.where(target_dx>23,1,0)
    forbidden_dy = jnp.where(target_dy<0,1,0) + jnp.where(target_dy>23,1,0)

    mask_dx = mask_dx + forbidden_dx
    mask_dy = mask_dy + forbidden_dy

    mask_dx = mask_dx.at[:,max_sap_range].set(0)
    mask_dy = mask_dy.at[:,max_sap_range].set(0)

    actor_dx = actor_dx - mask_dx*100
    actor_dy = actor_dy - mask_dy*100

    actor_dx = nn.log_softmax(actor_dx,axis=-1)
    actor_dy = nn.log_softmax(actor_dy,axis=-1)

    # Sampling action based on the policy
    action = jnp.zeros((n_units, 3), dtype=jnp.int32)

    action = action.at[:,0].set(action_choice)
    action_key, action_dx_key = jax.random.split(action_key)
    action = action.at[:,1].set(jax.random.categorical(key=action_dx_key, logits=actor_dx, axis=-1) - max_sap_range)
    action_key, action_dy_key = jax.random.split(action_key)
    action = action.at[:,2].set(jax.random.categorical(key=action_dy_key, logits=actor_dy, axis=-1) - max_sap_range)


    # Computing log probabilities for the actions
    unit_indices = jnp.arange(0,n_units)

    log_prob = actor_action[unit_indices, action[:,0]]+actor_dx[unit_indices, action[:,1]+max_sap_range]+actor_dy[unit_indices, action[:,2]+max_sap_range]

    return action, mask_action, mask_dx, mask_dy, log_prob

def compute_points(obs,previous_obs) :

    points_0 = jnp.clip(obs['player_0'].team_points[0] - previous_obs['player_0'].team_points[0],min=0)
    points_1 = jnp.clip(obs['player_1'].team_points[1] - previous_obs['player_1'].team_points[1],min=0)

    return points_0, points_1

def compute_distance(obs,target_dist,player) :
    player_id = jnp.where(player=='player_1',1,0)
    return jax.vmap(distance_to_target, in_axes=(0,None))(obs[player].units.position[player_id],target_dist)

def distance_to_target(unit,target_dist) :
    return target_dist[unit[0],unit[1]]

def compute_target_distance_matrix(state,obs,player) :
    target = jnp.where(state.relic_nodes_map_weights <= state.relic_nodes_mask.sum()//2,1,0)*jnp.where(state.relic_nodes_map_weights>0,1,0)
    
    player_idx = jnp.where(player=='player_1',1,0)
    units_mask = jnp.where(obs[player].units_mask[player_idx],0,1)
    unit_patch = jnp.ones((24,24),dtype=jnp.int32)
    for i, (x,y) in enumerate(obs[player].units.position[player_idx]) :
        unit_patch = unit_patch.at[x,y].set(units_mask[i])

    return min_manhattan_dist(target*unit_patch), target
        
def compute_reward(obs,previous_obs,action,new_distance,distance,target,player) :
    player_idx = jnp.where(player=='player_1',1,0)
    units_mask = jnp.where(obs[player].units_mask[player_idx],1,0)
    reward = -jnp.ones(16,dtype=jnp.int32) / 505
    delta_distance = new_distance - distance
    reward = reward - units_mask*(jnp.where(jnp.abs(delta_distance)>1,0,delta_distance)/(24*24)) * jnp.where(obs[player].match_steps==0,0,1)

    
    for i, (x,y) in enumerate(obs[player].units.position[player_idx]) :
            reward = reward.at[i].add(target[x,y]*units_mask[i]/20)
            target = target.at[x,y].set(1-units_mask[i])

    delta_energy = (obs[player].units.energy[player_idx] - previous_obs[player].units.energy[player_idx]) * jnp.where(previous_obs[player].units.energy[player_idx]>=0,1,0)
    reward = reward + units_mask * (delta_energy/(6000*jnp.log(jnp.clip(obs[player].units.energy[player_idx],min=2)))) * jnp.where(obs[player].match_steps==0,0,1)

    reward = reward - units_mask * jnp.where(obs[player].units.energy[player_idx]<0,1,0)/50

    allies_position = obs[player].units.position[player_idx]
    enemies_position = obs[player].units.position[1-player_idx]

    tx_min = jnp.repeat(jnp.expand_dims(jnp.clip(allies_position[:,0]+action[:,1]-1,min=0),axis=-1),16,axis=-1)
    tx_max = jnp.repeat(jnp.expand_dims(jnp.clip(allies_position[:,0]+action[:,1]+1,max=23),axis=-1),16,axis=-1)
    ty_min = jnp.repeat(jnp.expand_dims(jnp.clip(allies_position[:,1]+action[:,2]-1,min=0),axis=-1),16,axis=-1)
    ty_max = jnp.repeat(jnp.expand_dims(jnp.clip(allies_position[:,1]+action[:,2]+1,max=23),axis=-1),16,axis=-1)

    tx_touche = jnp.where(tx_min<=enemies_position[:,0],1,0) * jnp.where(enemies_position[:,0]<=tx_max,1,0)
    ty_touche = jnp.where(ty_min<=enemies_position[:,1],1,0) * jnp.where(enemies_position[:,1]<=ty_max,1,0)
    touche = tx_touche * ty_touche

    reward = reward + units_mask * jnp.sum(touche,axis=-1)*jnp.where(action[:,0]==5,1,0)*jnp.where(obs[player].units.energy[1-player_idx]<0,10,1) / 50

    return reward

def min_manhattan_dist(matrix):
    # Initialize distance matrix: 0 for ones, inf for others
    distances = jnp.where(matrix == 1, 0, 48)
    _, final_distances, _ = jax.lax.while_loop(condition, body, (distances, update(distances), 0))
    return final_distances

def update(distances):
        """Perform one iteration of distance propagation."""
        padded = jnp.pad(distances, ((1, 1), (1, 1)), constant_values=48)
        neighbors = [padded[1:-1, 2:], padded[1:-1, :-2], padded[2:, 1:-1], padded[:-2, 1:-1]]
        new_distances = jnp.minimum(distances, jnp.min(jnp.stack(neighbors), axis=0) + 1)
        return new_distances

# Iterate multiple times (at most h + w times for full spread)
def condition(val):
    old_dist, new_dist, _ = val
    return ~jnp.all(old_dist == new_dist)  # Stop when no change

def body(val):
    old_dist, new_dist, i = val
    updated = update(new_dist)
    return new_dist, updated, i + 1

def swap_action(action) :

    action = action.at[:,0].set(jnp.where(action[:,0] == 1, 2, jnp.where(action[:,0] == 2, 1, action[:,0])))
    action = action.at[:,0].set(jnp.where(action[:,0] == 3, 4, jnp.where(action[:,0] == 4, 3, action[:,0])))

    new_action_dx = -action[:,2]
    action = action.at[:,2].set(-action[:,1])
    action = action.at[:,1].set(new_action_dx)

    return action

def compute_advantage(value,episode_start,reward,final_episode_start,final_value,gamma,gae_lambda,n_steps,num_envs) :

    advantage = jnp.zeros((n_steps,2,num_envs,16),dtype=jnp.float32)  
    last_gae_lam = 0

    for step in reversed(range(n_steps)):

        if step == n_steps - 1 :
            next_value = final_value
            non_terminal = 1 - final_episode_start

        else:
            next_value = value[step + 1]
            non_terminal = 1 - episode_start[step + 1]

        delta = reward[step] + gamma * next_value * non_terminal - value[step]

        last_gae_lam = delta + gamma * gae_lambda * non_terminal * last_gae_lam

        advantage = advantage.at[step].set(last_gae_lam)
    
    return advantage


def obs_to_state_dict(obs,ep_params,points,map_memory,player):

    map_width = 24

    list_state_features = []
    state_maps = jnp.zeros((10, map_width, map_width), dtype=jnp.float32)

    if player == "player_0" :
        state_maps = state_maps.at[0].set(jnp.where(obs['sensor_mask'],1,0))  # sensor_mask
        state_maps = state_maps.at[1].set(obs['map_features']['energy'] / 20)  # map_energy
        state_maps = state_maps.at[2].set(obs['map_features']['tile_type'])  # map_tile_type
    else :
        state_maps = state_maps.at[0].set(jnp.flip(jnp.where(obs['sensor_mask'],1,0),axis=[-2,-1]).T)  # sensor_mask
        state_maps = state_maps.at[1].set(jnp.flip(obs['map_features']['energy'] / 20,axis=[-2,-1]).T)  # map_energy
        state_maps = state_maps.at[2].set(jnp.flip(obs['map_features']['tile_type'],axis=[-2,-1]).T)  # map_tile_type

    state_maps = state_maps.at[6].set(map_memory[6])
    
    relic_nodes_mask = jnp.where(obs['relic_nodes_mask'],1,0)
    for i, (x,y) in enumerate(obs['relic_nodes']) :
        state_maps = state_maps.at[6,x,y].add(relic_nodes_mask[i])
        state_maps = state_maps.at[6, 23-y,23-x].add(relic_nodes_mask[i])
    
    if player == 'player_0' :
        units_mask = jnp.where(obs['units_mask'][0],1,0)
        for i, (x,y) in enumerate(obs['units']['position'][0]) :
            state_maps = state_maps.at[x,y].set(units_mask[i])
    else :
        position_player_1 = obs['units']['position'][::-1,:,::-1]
        position_player_1 = jnp.where(position_player_1==-1,-1,23-position_player_1)
        units_mask = jnp.where(obs['units_mask'][1],1,0)
        for i, (x,y) in enumerate(position_player_1[0]) :
            state_maps = state_maps.at[x,y].set(units_mask[i])
    
    step_match = obs['match_steps']
    n_match = jnp.sum(obs['team_wins'])
    n_relic = jnp.sum(state_maps[6])

    # Compute memory map
    state_maps = state_maps.at[3].set(state_maps[0] + (1 - state_maps[0]) * jnp.flip(state_maps[0],axis=[-2,-1]).T)  # Because the map is symmetric
    state_maps = state_maps.at[4].set(state_maps[1] + (1 - state_maps[0]) * jnp.flip(state_maps[1],axis=[-2,-1]).T)
    state_maps = state_maps.at[5].set(state_maps[2] + (1 - state_maps[0]) * jnp.flip(state_maps[2],axis=[-2,-1]).T)

    begin_match = jnp.where(step_match==0,1,0)
    map_memory = map_memory.at[3].set(1-begin_match)

    find_all_relics = jnp.where(n_relic==6,1,0) + jnp.where(n_relic==((n_match + 1)*2),1,0)
    map_memory = jnp.clip(map_memory.at[3].add(find_all_relics),max=1)

    state_maps = state_maps.at[4].set(state_maps[4] + (1 - state_maps[3]) * map_memory[4])  # Add memory
    state_maps = state_maps.at[5].set(state_maps[5] + (1 - state_maps[3]) * map_memory[5])
    state_maps = state_maps.at[3].set(map_memory[3] + (1 - map_memory[3]) * state_maps[3])

    reset_point_prob = 1 - (jnp.where(n_match >= 3,1,0)*jnp.where(step_match > 0,1,0) + find_all_relics)

    map_memory = map_memory.at[9].set(jnp.where(map_memory[9] == 0, -2, map_memory[9])*reset_point_prob)
    map_relic = state_maps[6] + (1 - state_maps[0])*reset_point_prob + (1 - state_maps[3])*(1-reset_point_prob)

    state_maps = state_maps.at[8].set(jnp.clip(jax.scipy.signal.convolve2d(map_relic, jnp.ones((5, 5)), mode="same", boundary="fill", fillvalue=0), a_max=1))
    state_maps = state_maps.at[8].set(state_maps[8] * jnp.where(map_memory[9] != 0, 1, 0))

    question = state_maps[8] * state_maps[7]
    old_prob = map_memory[9] * state_maps[8]
    sure = jnp.sum(jnp.where(map_memory[9] * question == 1, 1, 0))
    sum_question = jnp.sum(question)

    points_bool = jnp.clip(jnp.where(points==0,1,0) + jnp.where(sum_question-sure==0,1,0))
    points_prob = (1-points_bool) * ((points - sure) / (sum_question-sure+points_bool))

    new_prob = question * points_prob + (1 - question) * (-1)

    state_maps = state_maps.at[9].set(jnp.where(new_prob > old_prob, new_prob, old_prob))
    rotated_map = jnp.flip(state_maps[9],axis=[-2,-1]).T
    state_maps = state_maps.at[9].set(jnp.where(state_maps[9] >= rotated_map, state_maps[9], rotated_map))

    
    if player == 'player_0' :
        # Units
        list_state_features.append(obs['units']['position'].flatten() / map_width)  # position
        list_state_features.append(obs['units']['energy'].flatten() / 400)  # energy
        list_state_features.append(jnp.where(obs['units_mask'],1,0).flatten())  # unit_mask

        # Game
        list_state_features.append(obs['team_points'].flatten() / 3000)  # team_points
        list_state_features.append(obs['team_wins'].flatten() / 5)  # team_wins

    else :
        # Units
        list_state_features.append(position_player_1.flatten() / map_width)  # position
        list_state_features.append(obs['units']['energy'][::-1].flatten() / 400)  # energy
        list_state_features.append(jnp.where(obs['units_mask'],1,0)[::-1].flatten())  # unit_mask

        # Game
        list_state_features.append(obs['team_points'][::-1].flatten() / 3000)  # team_points
        list_state_features.append(obs['team_wins'][::-1].flatten() / 5)  # team_wins

    list_state_features.append(obs['relic_nodes'].flatten() / map_width)  # relic_nodes
    list_state_features.append(jnp.where(obs['relic_nodes_mask'],1,0).flatten())  # relic_nodes_mask 

    list_state_features.append(obs['steps'].flatten() / 505)  # steps
    list_state_features.append(obs['match_steps'].flatten() / 101)  # match_steps

    list_state_features.append(jnp.array([ep_params['unit_move_cost']/6, 
                                          ep_params['unit_sap_cost']/51, 
                                          ep_params['unit_sap_range']/8, 
                                          ep_params['unit_sensor_range']/4], dtype=jnp.float32))  # Static information about the episode

    state_features = jnp.concatenate(list_state_features,dtype=jnp.float32)

    return state_maps, state_features

def compute_mask_actions_log_probs_dict(obs,ep_params,action_key,actor_action,actor_dx,actor_dy,player) :
    #obs = obs[player]
    player_id = jnp.where(player=='player_1',1,0)
    n_units = 16
    n_action = 6
    max_sap_range = 8

    if player=='player_0' :
        position = obs['units']['position'][player_id]
    else :
        position = obs['units']['position'][player_id,:,::-1]
        position = jnp.where(position==-1,-1,23-position)
    
    #Gather information for masking
    energy_mask = jnp.expand_dims(jnp.where(obs['units']['energy'][player_id]<ep_params['unit_move_cost'],1,0),axis=-1)
    sap_mask = jnp.where(obs['units']['energy'][player_id]<ep_params['unit_sap_cost'],1,0)

    #Compute action masks
    mask_action = jnp.zeros((n_units,n_action),dtype=jnp.int32)
    mask_dx = jnp.zeros((n_units,max_sap_range*2+1),dtype=jnp.int32)
    mask_dy = jnp.zeros((n_units,max_sap_range*2+1),dtype=jnp.int32)

    mask_action = mask_action.at[:,1:].add(energy_mask)
    mask_action = mask_action.at[:,-1].add(sap_mask)

    directions = jnp.array([[0,-1],[1,0],[0,1],[-1,0]])

    target_tiles = jnp.repeat(jnp.expand_dims(position,axis=-2),4,axis=-2) + directions

    clamp_target_tiles = jnp.clip(target_tiles,min=0,max=23).reshape(n_units*4,2)
    target_tiles_type = obs['map_features']['tile_type'][clamp_target_tiles[:,0],clamp_target_tiles[:,1]].reshape(n_units,4)

    out_board_tiles = jnp.where(target_tiles<0,1,0) + jnp.where(target_tiles>23,1,0)
    forbidden_move = out_board_tiles[:,:,0] + out_board_tiles[:,:,1] + jnp.where(target_tiles_type==2,1,0)
    mask_action = mask_action.at[:,1:-1].add(forbidden_move)

    actor_action = actor_action - mask_action*100
    actor_action = nn.log_softmax(actor_action,axis=-1)
    action_key, action_choice_key = jax.random.split(action_key)
    action_choice = jax.random.categorical(key=action_choice_key, logits=actor_action, axis=-1)

    sap_mask =  jnp.expand_dims(sap_mask+jnp.where(action_choice !=5,1,0),axis=-1)
    
    sap_range = jnp.arange(-max_sap_range,max_sap_range+1)
    forbidden_range = jnp.where(sap_range<-ep_params['unit_sap_range'],1,0) + jnp.where(ep_params['unit_sap_range']<sap_range,1,0)

    mask_dx = mask_dx.at[:].add(forbidden_range) 
    mask_dy = mask_dy.at[:].add(forbidden_range)

    mask_dx = mask_dx.at[:].add(sap_mask) 
    mask_dy = mask_dy.at[:].add(sap_mask)

    expand_postion = jnp.repeat(jnp.expand_dims(position,axis=-2),2*max_sap_range+1,axis=-2)
    target_dx = expand_postion[:,:,0] + sap_range
    target_dy = expand_postion[:,:,1] + sap_range

    forbidden_dx = jnp.where(target_dx<0,1,0) + jnp.where(target_dx>23,1,0)
    forbidden_dy = jnp.where(target_dy<0,1,0) + jnp.where(target_dy>23,1,0)

    mask_dx = mask_dx + forbidden_dx
    mask_dy = mask_dy + forbidden_dy

    mask_dx = mask_dx.at[:,max_sap_range].set(0)
    mask_dy = mask_dy.at[:,max_sap_range].set(0)

    actor_dx = actor_dx - mask_dx*100
    actor_dy = actor_dy - mask_dy*100

    actor_dx = nn.log_softmax(actor_dx,axis=-1)
    actor_dy = nn.log_softmax(actor_dy,axis=-1)

    # Sampling action based on the policy
    action = jnp.zeros((n_units, 3), dtype=jnp.int32)

    action = action.at[:,0].set(action_choice)
    action_key, action_dx_key = jax.random.split(action_key)
    action = action.at[:,1].set(jax.random.categorical(key=action_dx_key, logits=actor_dx, axis=-1) - max_sap_range)
    action_key, action_dy_key = jax.random.split(action_key)
    action = action.at[:,2].set(jax.random.categorical(key=action_dy_key, logits=actor_dy, axis=-1) - max_sap_range)


    # Computing log probabilities for the actions
    unit_indices = jnp.arange(0,n_units)

    log_prob = actor_action[unit_indices, action[:,0]]+actor_dx[unit_indices, action[:,1]+max_sap_range]+actor_dy[unit_indices, action[:,2]+max_sap_range]

    return action, mask_action, mask_dx, mask_dy, log_prob