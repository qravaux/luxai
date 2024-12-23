from src.luxai_s3.wrappers import LuxAIS3GymEnv
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.multiprocessing as mp
import numpy as np
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

def obs_to_state(obs:dict) -> torch.Tensor:

    n_units = len(obs['units']['position'][0])
    list_state = []

    #Units
    list_state.append(torch.from_numpy(obs['units']['position'].astype(float)).flatten()) #position
    list_state.append(torch.from_numpy(obs['units']['energy'].astype(float)).flatten()) #energy
    list_state.append(torch.from_numpy(obs['units_mask'].astype(float)).flatten()) #unit_mask
    
    #Map
    list_state.append(torch.from_numpy(obs['sensor_mask'].astype(float)).flatten()) #sensor_mask
    list_state.append(torch.from_numpy(obs['map_features']['energy'].astype(float)).flatten()) #map_energy
    list_state.append(torch.from_numpy(obs['map_features']['tile_type'].astype(float)).flatten()) #map_tile_type

    list_state.append(torch.from_numpy(obs['relic_nodes'].astype(float)).flatten()) #relic_nodes
    list_state.append(torch.from_numpy(obs['relic_nodes_mask'].astype(float)).flatten()) #relic_nodes_mask

    #Game
    list_state.append(torch.from_numpy(obs['team_points'].astype(float)).flatten()) #team_points
    list_state.append(torch.from_numpy(obs['team_wins'].astype(float)).flatten()) #team_wins

    list_state.append(torch.from_numpy(obs['steps'].astype(float)).flatten()) #steps
    list_state.append(torch.from_numpy(obs['match_steps'].astype(float)).flatten()) #match_steps
    
    return torch.cat(list_state)

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
    return action

class Policy(nn.Module) :

    def __init__(self,n_input,n_action,n_units,sap_range) :

        super(Policy,self).__init__()

        self.n_units = n_units
        self.n_action = n_action
        self.sap_range = sap_range

        self.inputs = nn.Linear(n_input,512,dtype=torch.double)

        self.hidden1 = nn.Linear(512,128,dtype=torch.double)
        self.hidden2 = nn.Linear(128,32,dtype=torch.double)

        self.actor_action = []
        self.actor_dx = []
        self.actor_dy = []

        self.critic_action = []
        self.critic_dx = []
        self.critic_dy = []

        for unit in range(self.n_units) :
            self.actor_action.append(nn.Linear(32,self.n_action,dtype=torch.double))
            self.actor_dx.append(nn.Linear(32,self.sap_range*2 + 1,dtype=torch.double))
            self.actor_dy.append(nn.Linear(32,self.sap_range*2 + 1,dtype=torch.double))

        self.critic = nn.Linear(32,1,dtype=torch.double)

    def forward(self,x) :

        x = F.relu(self.inputs(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))

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

    def __init__(self, worker_id, shared_queue, policy, victory_bonus=0) :
        super(Luxai_Worker, self).__init__()

        self.env = LuxAIS3GymEnv(numpy_output=True)
        self.n_units = self.env.env_params.max_units
        self.sap_range = self.env.env_params.unit_sap_range
        self.match_step = self.env.env_params.max_steps_in_match + 1
        self.len_episode = self.match_step * self.env.env_params.match_count_per_episode

        self.victory_bonus = victory_bonus
        self.n_inputs = 1880
        self.n_action = 6
        self.policy = policy
        self.shared_queue = shared_queue
        self.worker_id = worker_id

    def run(self) :

        while True:

            # Reset the environment and get the initial state
            obs, _ = self.env.reset()
            state_0 = obs_to_state(obs['player_0'])
            state_1 = obs_to_state(obs['player_1'])

            base_reward_0 = 0
            base_reward_1 = 0

            cumulated_rewards = torch.zeros(self.len_episode,2,dtype=torch.double)
            states = torch.zeros(self.len_episode,2,self.n_inputs,dtype=torch.double)
            actions = torch.zeros(self.len_episode,2,self.n_units,3,dtype=torch.int)

            for step in range(self.len_episode):

                #Compute actions probabilities and values
                actor_action_0, actor_dx_0, actor_dy_0, value_0 = self.policy(state_0)
                actor_action_1, actor_dx_1, actor_dy_1, value_1 = self.policy(state_1)

                # Sample actions based on probabilities
                action_0 = sample_action(actor_action_0, actor_dx_0, actor_dy_0, self.n_units, self.sap_range)
                action_1 = sample_action(actor_action_1, actor_dx_1, actor_dy_1, self.n_units, self.sap_range)

                # Take a step in the environment
                action = dict(player_0=np.array(action_0, dtype=int), player_1=np.array(action_1, dtype=int))
                obs, reward, truncated, done, info = self.env.step(action)

                # Process the reward and next state
                next_state_0 = obs_to_state(obs['player_0'])
                next_state_1 = obs_to_state(obs['player_1'])
                
                if step == 0 :
                    reward_memory = reward

                if reward['player_0'] > reward_memory['player_0'] or reward['player_1'] > reward_memory['player_1'] :
                    base_reward_0 = cumulated_rewards[step-1,0]
                    base_reward_1 = cumulated_rewards[step-1,1]

                    if reward['player_0'] > reward_memory['player_0'] :
                        base_reward_0 += self.victory_bonus
                    else : 
                        base_reward_1 += self.victory_bonus
                    reward_memory = reward
                
                cumulated_rewards[step,0] = obs['player_0']['team_points'][0] + base_reward_0
                cumulated_rewards[step,1] = obs['player_1']['team_points'][1] + base_reward_1
                
                states[step,0] = state_0
                states[step,1] = state_1

                actions[step,0] = action_0
                actions[step,1] = action_1

                state_0 = next_state_0
                state_1 = next_state_1

            rewards = cumulated_rewards - torch.cat((torch.zeros(1,2),cumulated_rewards[:-1]))
            self.shared_queue.put((states.detach(),actions.detach(),rewards.detach(),cumulated_rewards[-1].detach()))

if __name__ == "__main__":

    print('Initialise training environment...\n')
    lr = 1e-5
    batch_size = 101
    max_norm = 0.5
    entropy_coef = 0.01
    vf_coef = 0.5
    gamma = 0.99
    gae_lambda = 0.95
    save_dir = "policy"
    save_rate = 50

    n_epochs = int(1e6)
    num_workers = 4
    n_episode = num_workers
     
    env = LuxAIS3GymEnv(numpy_output=True)
    n_units = env.env_params.max_units
    sap_range = env.env_params.unit_sap_range
    match_step = env.env_params.max_steps_in_match + 1
    len_episode = match_step * env.env_params.match_count_per_episode
    n_action = 6
    n_input = 1880
    step_cpt = 0
    
    writer = SummaryWriter("runs/experiment_1")

    model = Policy(n_input, n_action, n_units, sap_range=sap_range)
    model.share_memory()  # For multiprocessing, the model parameters must be shared
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    shared_queue = mp.Queue()  # Queue to share data between workers and the main process

    print('Instantiate workers...')
    workers = []
    for i in range(num_workers):
        worker = Luxai_Worker(i, shared_queue, model)
        workers.append(worker)
        worker.start()
    print('Done\n')
    
    # Main training loop: Collect experiences from workers and update the model
    print('Start training...\n')
    for epoch in range(n_epochs):

        # Collect data from workers
        experiences = []
        for _ in range(n_episode):

            states, actions, rewards, cumulated_rewards = shared_queue.get()

            #Compute log_probs and values

            actor_action, actor_dx, actor_dy, values = model(states.flatten(0,1))
            log_probs = compute_log_prob(actor_action, actor_dx, actor_dy, actions.flatten(0,1))

            # Advantage computation
            values = values.view(len_episode,2)
            advantages = torch.zeros(len_episode,2)
            
            last_gae_lam = torch.zeros(2)
            for step in reversed(range(len_episode)):

                if step == len_episode - 1 :
                    next_values = values[-1]
                else:
                    next_values = values[step + 1]

                delta = rewards[step] + gamma * next_values - values[step]
                last_gae_lam = delta + gamma * gae_lambda * last_gae_lam
                advantages[step] = last_gae_lam

            returns = advantages + values  
            returns = returns.flatten(0,1)
            advantages = advantages.flatten(0,1)
            values = values.flatten(0,1)

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            experiences.append((values, log_probs, advantages, returns, cumulated_rewards))

        # Process the collected experiences
        values, log_probs, advantages, returns, cumulated_rewards = zip(*experiences)

        values = torch.cat(values,dim=0)
        log_probs = torch.cat(log_probs,dim=0)
        advantages = torch.cat(advantages,dim=0)
        returns = torch.cat(returns,dim=0)
        cumulated_rewards = torch.stack(cumulated_rewards).type(torch.DoubleTensor)

        # Losses

        entropy_loss = -torch.mean(torch.exp(log_probs) * log_probs)
        policy_loss = -torch.mean(log_probs * advantages)
        value_loss = F.mse_loss(values, returns)

        loss = policy_loss + vf_coef *value_loss + entropy_coef * entropy_loss

        # Update model
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()

        step_cpt += len_episode*n_episode

        writer.add_scalar("Loss/Total Loss", loss.item(), step_cpt)
        writer.add_scalar("Loss/Policy Loss", policy_loss.item(), step_cpt)
        writer.add_scalar("Loss/Value Loss", value_loss.item(), step_cpt)
        writer.add_scalar("Loss/Entropy Loss", entropy_loss.item(), step_cpt)
        
        writer.add_scalar("Reward agent 0", torch.mean(cumulated_rewards[:,0]).item(), step_cpt)
        writer.add_scalar("Reward agent 1", torch.mean(cumulated_rewards[:,1]).item(), step_cpt)

        print(f"Episode {epoch}, Loss: {loss.item()}, Reward_0 : {torch.mean(cumulated_rewards[:,0]).item()}, Reward_1 : {torch.mean(cumulated_rewards[:,1]).item()}")

        if epoch % save_rate == 0:
            save_path = os.path.join(save_dir, f"policy_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)

    # Terminate workers
    for worker in workers:
        worker.terminate()
    
    writer.close()