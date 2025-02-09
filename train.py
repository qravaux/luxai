from src.luxai_s3.wrappers import LuxAIS3GymEnv
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.multiprocessing as mp
import numpy as np
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

def obs_to_state(obs:dict) -> torch.Tensor:
    list_state = []

    #Units
    list_state.append(torch.from_numpy(obs['units']['position'].astype(float)).flatten()/24) #position
    list_state.append(torch.from_numpy(obs['units']['energy'].astype(float)).flatten()/400) #energy
    list_state.append(torch.from_numpy(obs['units_mask'].astype(float)).flatten()) #unit_mask
    
    #Map
    list_state.append(torch.from_numpy(obs['sensor_mask'].astype(float)).flatten()) #sensor_mask
    list_state.append(torch.from_numpy(obs['map_features']['energy'].astype(float)).flatten()) #map_energy
    list_state.append(torch.from_numpy(obs['map_features']['tile_type'].astype(float)).flatten()) #map_tile_type

    list_state.append(torch.from_numpy(obs['relic_nodes'].astype(float)).flatten()) #relic_nodes
    list_state.append(torch.from_numpy(obs['relic_nodes_mask'].astype(float)).flatten()) #relic_nodes_mask

    #Game
    list_state.append(torch.from_numpy(obs['team_points'].astype(float)).flatten()/3000) #team_points
    list_state.append(torch.from_numpy(obs['team_wins'].astype(float)).flatten()/5) #team_wins

    list_state.append(torch.from_numpy(obs['steps'].astype(float)).flatten()/100) #steps
    list_state.append(torch.from_numpy(obs['match_steps'].astype(float)).flatten()/5) #match_steps
    
    state = torch.cat(list_state)

    return state

def compute_log_prob(actor_action, actor_dx, actor_dy, action):
    batch_size = actor_action.size(0)
    n_units = actor_action.size(1)

    step_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, n_units)
    unit_indices = torch.arange(n_units).unsqueeze(0).expand(batch_size, -1) 

    # Computing log probabilities for the actions
    log_prob = torch.sum(actor_action[step_indices,unit_indices, action[:,:, 0]],axis=1)
    log_prob += torch.sum(actor_dx[step_indices,unit_indices, action[:,:, 1]],axis=1)
    log_prob += torch.sum(actor_dy[step_indices,unit_indices, action[:,:, 2]],axis=1)

    return log_prob

def sample_action(actor_action, actor_dx, actor_dy, n_units, sap_range):

    # Sampling action based on the policy
    action = torch.zeros(n_units, 3, dtype=torch.int)
    action[:, 0] = Categorical(logits=actor_action[0]).sample()
    action[:, 1] = Categorical(logits=actor_dx[0]).sample() - sap_range
    action[:, 2] = Categorical(logits=actor_dy[0]).sample() - sap_range

    sap = (action[:,0] != 5)
    action[sap,1] = 0
    action[sap,2] = 0

    return action

class Policy(nn.Module) :

    def __init__(self,n_input,n_action,n_units,sap_range) :

        super(Policy,self).__init__()

        self.n_units = n_units
        self.n_action = n_action
        self.sap_range = sap_range

        self.inputs = nn.Linear(n_input,64,dtype=torch.double)

        #self.hidden1 = nn.Linear(1024,512,dtype=torch.double)
        #self.hidden2 = nn.Linear(512,5,dtype=torch.double)

        self.actor_action = []
        self.actor_dx = []
        self.actor_dy = []

        for unit in range(self.n_units) :
            self.actor_action.append(nn.Linear(64,self.n_action,dtype=torch.double))
            self.actor_dx.append(nn.Linear(64,self.sap_range*2 + 1,dtype=torch.double))
            self.actor_dy.append(nn.Linear(64,self.sap_range*2 + 1,dtype=torch.double))

        self.critic = nn.Linear(64,1,dtype=torch.double)

    def forward(self,x) :

        x = F.relu(self.inputs(x))
        #x = F.relu(self.hidden1(x))
        #x = F.relu(self.hidden2(x))

        actor_action = torch.zeros(x.size(0),self.n_units,self.n_action)
        actor_dx = torch.zeros(x.size(0),self.n_units,self.sap_range*2 + 1)
        actor_dy = torch.zeros(x.size(0),self.n_units,self.sap_range*2 + 1)

        for unit in range(self.n_units) :
            actor_action[:,unit] = F.log_softmax(self.actor_action[unit](x),dim=-1)
            actor_dx[:,unit] = F.log_softmax(self.actor_dx[unit](x),dim=-1)
            actor_dy[:,unit] = F.log_softmax(self.actor_dy[unit](x),dim=-1)

        value = self.critic(x)

        return actor_action,actor_dx,actor_dy,value
    
class Luxai_Worker(mp.Process) :

    def __init__(self, worker_id, shared_queue, policy_0, policy_1, victory_bonus, gamma, gae_lambda, n_steps,reward_queue, event) :

        super(Luxai_Worker, self).__init__()

        self.env = LuxAIS3GymEnv(numpy_output=True)
        self.n_units = self.env.env_params.max_units
        self.sap_range = self.env.env_params.unit_sap_range
        self.match_step = self.env.env_params.max_steps_in_match + 1
        self.len_episode = self.match_step * self.env.env_params.match_count_per_episode
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_steps = n_steps

        self.victory_bonus = victory_bonus
        self.n_inputs = 1880
        self.n_action = 6

        self.policy_0 = policy_0
        self.policy_1 = policy_1

        self.shared_queue = shared_queue
        self.reward_queue = reward_queue
        self.event = event
        self.worker_id = worker_id

        for _ in range(self.worker_id):#*8) :
            _, _ = self.env.reset()

    def run(self) :

        with torch.no_grad():

            states = torch.zeros(2,self.n_steps,self.n_inputs,dtype=torch.double)
            actions = torch.zeros(2,self.n_steps,self.n_units,3,dtype=torch.int)
            values = torch.zeros(2,self.n_steps,1,dtype=torch.double)
            rewards = torch.zeros(2,self.n_steps,dtype=torch.double)
            episode_start = torch.zeros(self.n_steps,dtype=torch.double)

            step_cpt = 0

            while True:

                # Reset the environment and get the initial state
                obs, _ = self.env.reset()
                state_0 = obs_to_state(obs['player_0'])
                state_1 = obs_to_state(obs['player_1'])
                previous_obs = obs
                
                cumulated_reward = np.zeros(2)

                for ep_step in range(self.len_episode):

                    #Compute actions probabilities and values
                    actor_action_0, actor_dx_0, actor_dy_0, value_0 = self.policy_0(state_0)
                    actor_action_1, actor_dx_1, actor_dy_1, value_1 = self.policy_1(state_1)

                    # Sample actions based on probabilities
                    action_0 = sample_action(actor_action_0, actor_dx_0, actor_dy_0, self.n_units, self.sap_range)
                    action_1 = sample_action(actor_action_1, actor_dx_1, actor_dy_1, self.n_units, self.sap_range)

                    # Take a step in the environment
                    action = dict(player_0=np.array(action_0, dtype=int), player_1=np.array(action_1, dtype=int))
                    obs, reward, truncated, done, info = self.env.step(action)

                    # If the Buffer is full, send collected trajectories to the Queue
                    if step_cpt == self.n_steps :
                        
                        # Advantage computation
                        advantages = torch.zeros(2,self.n_steps,1)                    
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

                        advantages[0] = (advantages[0] - advantages[0].mean()) / (advantages[0].std() + 1e-8)
                        advantages[1] = (advantages[1] - advantages[1].mean()) / (advantages[1].std() + 1e-8)

                        self.shared_queue.put((states,actions,advantages,returns))

                        step_cpt = 0
                        states = torch.zeros(2,self.n_steps,self.n_inputs,dtype=torch.double)
                        actions = torch.zeros(2,self.n_steps,self.n_units,3,dtype=torch.int)
                        values = torch.zeros(2,self.n_steps,1,dtype=torch.double)
                        rewards = torch.zeros(2,self.n_steps,dtype=torch.double)
                        episode_start = torch.zeros(self.n_steps,dtype=torch.double)

                        self.event.wait()
                        self.event.clear()

                    # Compute the rewards
                    next_state_0 = obs_to_state(obs['player_0'])
                    next_state_1 = obs_to_state(obs['player_1'])
                    episode_start[step_cpt] = 0
                    
                    if ep_step == 0 :
                        episode_start[step_cpt] = 1
                        reward_memory = reward
                        rewards[0,step_cpt] = obs['player_0']['team_points'][0]
                        rewards[1,step_cpt] = obs['player_1']['team_points'][1]

                    elif reward['player_0'] > reward_memory['player_0'] :
                        reward_memory = reward
                        rewards[0,step_cpt] = obs['player_0']['team_points'][0] + self.victory_bonus
                        rewards[1,step_cpt] = obs['player_1']['team_points'][1]

                    elif reward['player_1'] > reward_memory['player_1'] :
                        reward_memory = reward
                        rewards[0,step_cpt] = obs['player_0']['team_points'][0]
                        rewards[1,step_cpt] = obs['player_1']['team_points'][1] + self.victory_bonus

                    else :
                        rewards[0,step_cpt] = obs['player_0']['team_points'][0] - previous_obs['player_0']['team_points'][0]
                        rewards[1,step_cpt] = obs['player_1']['team_points'][1] - previous_obs['player_1']['team_points'][1]
                    
                    cumulated_reward[0] += rewards[0,step_cpt]
                    cumulated_reward[1] += rewards[1,step_cpt]

                    #Update the trajectories
                    states[0,step_cpt] = state_0
                    states[1,step_cpt] = state_1

                    actions[0,step_cpt] = action_0
                    actions[1,step_cpt] = action_1

                    values[0,step_cpt] = value_0
                    values[1,step_cpt] = value_1

                    state_0 = next_state_0
                    state_1 = next_state_1

                    step_cpt += 1
                    previous_obs = obs

                self.reward_queue.put(cumulated_reward)


class ReplayBuffer(Dataset):
    def __init__(self,states,actions,advantages,returns):

        self.states = states
        self.actions = actions
        self.advantages = advantages
        self.returns = returns

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.advantages[idx], self.returns[idx]
    
if __name__ == "__main__":

    print('Initialise training environment...\n')
    lr0 = 1e-5
    lr1 = 1e-6
    max_norm0 = 1.0
    max_norm1 = 0.5

    batch_size = 50
    entropy_coef = 0.1
    vf_coef = 1.0
    gamma = 0.99
    gae_lambda = 0.95
    save_dir = "policy"
    save_rate = 100

    n_epochs = int(1e6)
    n_batch = 10
    num_workers = 6
    n_episode = num_workers
    n_steps = 100
     
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
    
    writer = SummaryWriter("runs/experiment_1")

    model_0 = Policy(n_input, n_action, n_units, sap_range=sap_range)
    model_0.share_memory()  # For multiprocessing, the model parameters must be shared
    optimizer_0 = torch.optim.Adam(model_0.parameters(), lr=lr0)

    model_1 = Policy(n_input, n_action, n_units, sap_range=sap_range)
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

            states,actions,advantages,returns = shared_queue.get()

            experiences_0.append((states[0],actions[0],advantages[0],returns[0]))
            experiences_1.append((states[1],actions[1],advantages[1],returns[1]))

        # Process the collected experiences
        states_0,actions_0,advantages_0,returns_0 = zip(*experiences_0)
        states_1,actions_1,advantages_1,returns_1 = zip(*experiences_1)

        states_0 = torch.cat(states_0,dim=0)
        actions_0 = torch.cat(actions_0,dim=0)
        advantages_0 = torch.cat(advantages_0,dim=0)
        returns_0 = torch.cat(returns_0,dim=0)

        states_1 = torch.cat(states_1,dim=0)
        actions_1 = torch.cat(actions_1,dim=0)
        advantages_1 = torch.cat(advantages_1,dim=0)
        returns_1 = torch.cat(returns_1,dim=0)

        train_data_0 = ReplayBuffer(states_0,actions_0,advantages_0,returns_0)
        train_data_1 = ReplayBuffer(states_1,actions_1,advantages_1,returns_1)
 
        train_loader_0 = DataLoader(train_data_0, batch_size=batch_size, shuffle=True)
        train_loader_1 = DataLoader(train_data_1, batch_size=batch_size, shuffle=True)

        for _ in range(n_batch) :

            for batch in train_loader_0 :

                states_,actions_,advantages_,returns_ = batch

                #Compute log_probs and values
                actor_action_, actor_dx_, actor_dy_, values_ = model_0(states_)
                log_probs_ = compute_log_prob(actor_action_, actor_dx_, actor_dy_, actions_)

                # Losses
                entropy_loss_0 = -torch.mean(torch.exp(log_probs_) * log_probs_)
                policy_loss_0 = -torch.mean(log_probs_ * advantages_)
                value_loss_0 = F.mse_loss(values_, returns_)

                loss_0 = policy_loss_0 + vf_coef *value_loss_0 + entropy_coef * entropy_loss_0

                # Update model
                optimizer_0.zero_grad()
                loss_0.backward()
                torch.nn.utils.clip_grad_norm_(model_0.parameters(), max_norm=max_norm0)
                optimizer_0.step()

            for batch in train_loader_1 :

                states_,actions_,advantages_,returns_ = batch

                #Compute log_probs and values
                actor_action_, actor_dx_, actor_dy_, values_ = model_1(states_)
                log_probs_ = compute_log_prob(actor_action_, actor_dx_, actor_dy_, actions_)

                # Losses
                entropy_loss_1 = -torch.mean(torch.exp(log_probs_) * log_probs_)
                policy_loss_1 = -torch.mean(log_probs_ * advantages_)
                value_loss_1 = F.mse_loss(values_, returns_)

                loss_1 = policy_loss_1 + vf_coef *value_loss_1 + entropy_coef * entropy_loss_1

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

        step_cpt += n_episode*n_steps

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

        event.set()

        if epoch % save_rate == 0 :
            save_path = os.path.join(save_dir, f"policy_0_epoch_{epoch}.pth")
            torch.save(model_0.state_dict(), save_path)

            save_path = os.path.join(save_dir, f"policy_1_epoch_{epoch}.pth")
            torch.save(model_1.state_dict(), save_path)

    # Terminate workers
    for worker in workers :
        worker.join()
    
    writer.close()