import gymnasium as gym
from src.luxai_s3.wrappers import LuxAIS3GymEnv
from gym.spaces import Dict, Box, MultiDiscrete
import numpy as np

class luxai_agent(gym.Env):

    def __init__(self) :
        self.env = LuxAIS3GymEnv(numpy_output=True)
        self.observation_space = gym.spaces.Dict("player0" : )

    def obs_to_state(self,obs) :

        n_units = len(obs['units']['position'][0])
        list_state = []

        #Units
        list_state.append(obs['units']['position'].astype(float)).flatten() #position
        list_state.append(obs['units']['energy'].astype(float)).flatten() #energy
        list_state.append(obs['units_mask'].astype(float)).flatten() #unit_mask
        
        #Map
        list_state.append(obs['sensor_mask'].astype(float)).flatten() #sensor_mask
        list_state.append(obs['map_features']['energy'].astype(float)).flatten() #map_energy
        list_state.append(obs['map_features']['tile_type'].astype(float)).flatten() #map_tile_type

        list_state.append(obs['relic_nodes'].astype(float)).flatten() #relic_nodes
        list_state.append(obs['relic_nodes_mask'].astype(float)).flatten() #relic_nodes_mask

        #Game
        list_state.append(obs['team_points'].astype(float)).flatten() #team_points
        list_state.append(obs['team_wins'].astype(float)).flatten() #team_wins

        list_state.append(obs['steps'].astype(float)).flatten() #steps
        list_state.append(obs['match_steps'].astype(float)).flatten() #match_steps
        
        return np.cat(list_state)

    def reset(self) :

        obs, _ = self.env.reset()
        state_0 = self.obs_to_state(obs['player_0'])
        state_1 = self.obs_to_state(obs['player_1'])

        base_reward_0 = 0
        base_reward_1 = 1