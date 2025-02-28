import sys
sys.path.append('../../')
from lux.utils import direction_to
from lux.kit import to_numpy
import numpy as np
from policies import Luxai_Agent
import flax.serialization
import torch

class Agent():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg

        self.ep_params = {}
        self.ep_params['unit_move_cost'] = env_cfg['unit_move_cost']
        self.ep_params['unit_sap_cost'] = env_cfg['unit_sap_cost']
        self.ep_params['unit_sap_range'] = env_cfg['unit_sap_range']
        self.ep_params['unit_sensor_range'] = env_cfg['unit_sensor_range']

        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.unit_explore_locations = dict()

        self.policy_0 = Luxai_Agent(self.player)
        self.policy_1 = Luxai_Agent(self.player)

        self.policy_0.load_state_dict(torch.load("../../policy/PPO_unit_reward/policy_0_epoch_0.pth", weights_only=True))
        self.policy_1.load_state_dict(torch.load("../../policy/PPO_unit_reward/policy_1_epoch_0.pth", weights_only=True))

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        good_obs = to_numpy(flax.serialization.to_state_dict(obs))

        if self.player == "player_0" :

            if step == 0 :
                points_0 = obs['team_points'][0]
                self.previous_obs = obs
                state_maps ,state_features = self.policy_0.obs_to_state(good_obs,self.ep_params,points_0)
                self.map_memory_0 = state_maps
                
            else :
                points_0 = (obs['team_points'][0] - self.previous_obs['team_points'][0])
                self.previous_obs = obs
                state_maps ,state_features = self.policy_0.obs_to_state(good_obs,self.ep_params,points_0,self.map_memory_0)
                self.map_memory_0 = state_maps

            action,_,_,_,_,_ = self.policy_0(state_maps ,state_features,good_obs,self.ep_params)
            return action.numpy()
        elif self.player == "player_1" :
            if step == 0 :
                points_1 = obs['team_points'][1]
                self.previous_obs = obs
                state_maps ,state_features = self.policy_1.obs_to_state(good_obs,self.ep_params,points_1)
                self.map_memory_1 = state_maps
                
            else :
                points_1 = (obs['team_points'][1] - self.previous_obs['team_points'][1])
                self.previous_obs = obs
                state_maps ,state_features = self.policy_1.obs_to_state(good_obs,self.ep_params,points_1,self.map_memory_1)
                self.map_memory_1 = state_maps

            action,_,_,_,_,_ = self.policy_1(state_maps ,state_features,good_obs,self.ep_params)
            return action.numpy()


        """
        unit_mask = np.array(obs["units_mask"][self.team_id]) # shape (max_units, )
        unit_positions = np.array(obs["units"]["position"][self.team_id]) # shape (max_units, 2)
        unit_energys = np.array(obs["units"]["energy"][self.team_id]) # shape (max_units, 1)
        observed_relic_node_positions = np.array(obs["relic_nodes"]) # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"]) # shape (max_relic_nodes, )
        team_points = np.array(obs["team_points"]) # points of each team, team_points[self.team_id] is the points of the your team
        
        # ids of units you can control at this timestep
        available_unit_ids = np.where(unit_mask)[0]
        # visible relic nodes
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])
        
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)


        # basic strategy here is simply to have some units randomly explore and some units collecting as much energy as possible
        # and once a relic node is found, we send all units to move randomly around the first relic node to gain points
        # and information about where relic nodes are found are saved for the next match
        
        # save any new relic nodes that we discover for the rest of the game.
        for id in visible_relic_node_ids:
            if id not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(id)
                self.relic_node_positions.append(observed_relic_node_positions[id])
            

        # unit ids range from 0 to max_units - 1
        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]
            unit_energy = unit_energys[unit_id]
            if len(self.relic_node_positions) > 0:
                nearest_relic_node_position = self.relic_node_positions[0]
                manhattan_distance = abs(unit_pos[0] - nearest_relic_node_position[0]) + abs(unit_pos[1] - nearest_relic_node_position[1])
                
                # if close to the relic node we want to hover around it and hope to gain points
                if manhattan_distance <= 4:
                    random_direction = np.random.randint(0, 5)
                    actions[unit_id] = [random_direction, 0, 0]
                else:
                    # otherwise we want to move towards the relic node
                    actions[unit_id] = [direction_to(unit_pos, nearest_relic_node_position), 0, 0]
            else:
                # randomly explore by picking a random location on the map and moving there for about 20 steps
                if step % 20 == 0 or unit_id not in self.unit_explore_locations:
                    rand_loc = (np.random.randint(0, self.env_cfg["map_width"]), np.random.randint(0, self.env_cfg["map_height"]))
                    self.unit_explore_locations[unit_id] = rand_loc
                actions[unit_id] = [direction_to(unit_pos, self.unit_explore_locations[unit_id]), 0, 0]

        return actions
        """
        
