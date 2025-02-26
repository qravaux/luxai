import numpy as np
from lux.policies import Luxai_Agent
from lux.utils import direction_to
from lux.kit import to_numpy
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

        self.policy_0.load_state_dict(torch.load("policy/policy_0_epoch_180.pth", weights_only=True))
        self.policy_1.load_state_dict(torch.load("policy/policy_1_epoch_180.pth", weights_only=True))

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        good_obs = to_numpy(flax.serialization.to_state_dict(obs))

        if self.player == "player_0" :

            if step == 0 :
                state_maps ,state_features = self.policy_0.obs_to_state(good_obs,self.ep_params)
                self.map_memory_0 = state_maps[3:]
            else :
                state_maps ,state_features = self.policy_0.obs_to_state(good_obs,self.ep_params,self.map_memory_0)
                self.map_memory_0 = state_maps[3:]

            action,_,_,_,_,_ = self.policy_0(state_maps ,state_features,good_obs,self.ep_params)
            return action.numpy()
        else :
            if step == 0 :
                state_maps ,state_features = self.policy_1.obs_to_state(good_obs,self.ep_params)
                self.map_memory_1 = state_maps[3:]
            else :
                state_maps ,state_features = self.policy_1.obs_to_state(good_obs,self.ep_params,self.map_memory_1)
                self.map_memory_1 = state_maps[3:]

            action,_,_,_,_,_ = self.policy_1(state_maps ,state_features,good_obs,self.ep_params)
            return action.numpy()