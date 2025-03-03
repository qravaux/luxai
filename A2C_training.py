import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch import nn

from src.luxai_s3.wrappers import LuxAIS3GymEnv
from policies import Luxai_Agent
from workers import Luxai_Worker

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
          
if __name__ == "__main__":

    print('Initialise training environment...\n')
    lr0 = 1e-8
    lr1 = 1e-9
    max_norm0 = 0.5
    max_norm1 = 0.5
    entropy_coef0 = 0.001
    entropy_coef1 = 0.001
    weight_decay_0 = 0
    weight_decay_1 = 0
    eps = 1e-8
    betas = (0.9,0.999)

    replay_buffer_capacity = 50000
    batch_size = 100
    vf_coef = 0.5
    gamma = 0.995
    gae_lambda = 0.99
    save_rate = 1000

    n_epochs = int(1e6)
    n_batch = 10
    num_workers = 6
    n_episode = 3
    n_steps = batch_size // 2

    file_name = 'A2C_unit_reward'
    save_dir = f"policy/{file_name}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    model_0 = Luxai_Agent('player_0')
    model_0_gpu = Luxai_Agent('player_0')
    model_0_gpu.load_state_dict(model_0.state_dict())
    model_0_gpu.cuda(device)
    optimizer_0 = torch.optim.Adam(model_0_gpu.parameters(), lr=lr0, weight_decay=weight_decay_0, eps=eps, betas=betas)
    model_0.share_memory()  # For multiprocessing, the model parameters must be shared

    model_1 = Luxai_Agent('player_1')
    model_1_gpu = Luxai_Agent('player_1')
    model_1_gpu.load_state_dict(model_1.state_dict())
    model_1_gpu.cuda(device)
    optimizer_1 = torch.optim.Adam(model_1_gpu.parameters(), lr=lr1, weight_decay=weight_decay_1, eps=eps, betas=betas)
    model_1.share_memory()  # For multiprocessing, the model parameters must be shared

    shared_queue = mp.Queue(maxsize=n_batch * (batch_size//n_steps))  # Queue to share data between workers and the main process
    reward_queue = mp.Queue(maxsize=n_batch * (batch_size//n_steps))
    point_queue = mp.Queue(maxsize=n_batch * (batch_size//n_steps))
    event = mp.Event()

    replay_buffer_0 = PrioritizedReplayBuffer(capacity=replay_buffer_capacity,device=device)
    replay_buffer_1 = PrioritizedReplayBuffer(capacity=replay_buffer_capacity,device=device)

    print('Instantiate workers...')
    workers = []
    for i in range(num_workers) :
        worker = Luxai_Worker(i,shared_queue,   
                              reward_queue,
                              point_queue,
                              event,
                              model_0,
                              model_1,
                              gamma,
                              gae_lambda,
                              n_steps,
                              n_episode,
                              n_epochs,
                              victory_bonus,)
        workers.append(worker)
        worker.start()
        print(f'--------worker {i} is ready')
    print('Done\n')
    
    while shared_queue.qsize() < n_batch * (batch_size//n_steps) :
        waiting = True

    # Main training loop: Collect experiences from workers and update the model
    print('Start training...\n')
    for epoch in range(n_epochs):

        # Collect data from workers
        while shared_queue.qsize() > 0 :
            states_maps,states_features,actions,advantages,returns,log_probs,mask_actions,mask_dxs,mask_dys = shared_queue.get()

            replay_buffer_0.add(states_maps[0],
                                states_features[0],
                                actions[0],
                                advantages[0],
                                returns[0],
                                log_probs[0],
                                mask_actions[0],
                                mask_dxs[0],
                                mask_dys[0])
            
            replay_buffer_1.add(states_maps[1],
                                states_features[1],
                                actions[1],
                                advantages[1],
                                returns[1],
                                log_probs[1],
                                mask_actions[1],
                                mask_dxs[1],
                                mask_dys[1])
            del states_maps,states_features,actions,advantages,returns,log_probs,mask_actions,mask_dxs,mask_dys

        while reward_queue.qsize() > 0 :
            reward = reward_queue.get_nowait()
            point = point_queue.get_nowait()
            reward_cpt += len_episode

            writer.add_scalar("Reward 0", reward[0].item(), reward_cpt)
            writer.add_scalar("Reward 1", reward[1].item(), reward_cpt)

            writer.add_scalar("Point 0", point[0].item(), reward_cpt)
            writer.add_scalar("Point 1", point[1].item(), reward_cpt)

        for _ in range(n_batch) :

            states_maps_, states_features_, actions_, advantages_, returns_, p_log_probs_, mask_actions_, mask_dxs_, mask_dys_, weights_, indices_ = replay_buffer_0.sample(batch_size=batch_size,device=device)

            #Compute log_probs and values
            values_,log_probs_ = model_0_gpu.training_forward(states_maps_,states_features_,actions_,mask_actions_,mask_dxs_,mask_dys_)
            advantages_ = (advantages_ - torch.mean(advantages_,dim=0)) / (torch.std(advantages_,dim=0) + 1e-8)
        
            # Losses
            entropy_loss_0 = -torch.mean(weights_*torch.sum(torch.exp(log_probs_) * log_probs_,dim=-1))
            policy_loss_0 = -torch.mean(weights_*torch.sum(log_probs_ * advantages_,dim=-1))
            value_loss_0 = torch.mean(weights_*torch.sum(torch.pow(values_ - returns_,2),dim=-1))

            loss_0 = policy_loss_0 + vf_coef *value_loss_0 + entropy_coef0 * entropy_loss_0

            # Update model
            optimizer_0.zero_grad()
            loss_0.backward()
            torch.nn.utils.clip_grad_norm_(model_0_gpu.parameters(), max_norm=max_norm0)
            optimizer_0.step()

            replay_buffer_0.update_priorities(batch_indices=indices_,batch_priorities=torch.abs(torch.sum(values_,dim=-1)).detach())

            states_maps_, states_features_, actions_, advantages_, returns_, p_log_probs_, mask_actions_, mask_dxs_, mask_dys_, weights_, indices_ = replay_buffer_1.sample(batch_size=batch_size,device=device)

            #Compute log_probs and values
            values_,log_probs_ = model_1_gpu.training_forward(states_maps_,states_features_,actions_,mask_actions_,mask_dxs_,mask_dys_)
            advantages_ = (advantages_ - torch.mean(advantages_,dim=0)) / (torch.std(advantages_,dim=0) + 1e-8)

            # Losses
            entropy_loss_1 = -torch.mean(weights_*torch.sum(torch.exp(log_probs_) * log_probs_,dim=-1))
            policy_loss_1 = -torch.mean(weights_*torch.sum(log_probs_ * advantages_,dim=-1))
            value_loss_1 = torch.mean(weights_*torch.sum(torch.pow(values_ - returns_,2),dim=-1))

            loss_1 = policy_loss_1 + vf_coef *value_loss_1 + entropy_coef1 * entropy_loss_1

            # Update model
            optimizer_1.zero_grad()
            loss_1.backward()
            torch.nn.utils.clip_grad_norm_(model_1_gpu.parameters(), max_norm=max_norm1)
            optimizer_1.step()

            replay_buffer_1.update_priorities(batch_indices=indices_,batch_priorities=torch.abs(torch.sum(values_,dim=-1)).detach())

            step_cpt += batch_size

        model_0.load_state_dict(model_0_gpu.state_dict())
        model_1.load_state_dict(model_1_gpu.state_dict())

        writer.add_scalar("Loss/Policy Loss 0", policy_loss_0.cpu().item(), step_cpt)
        writer.add_scalar("Loss/Policy Loss 1", policy_loss_1.cpu().item(), step_cpt)

        writer.add_scalar("Loss/Value Loss 0", value_loss_0.cpu().item(), step_cpt)
        writer.add_scalar("Loss/Value Loss 1", value_loss_1.cpu().item(), step_cpt)

        writer.add_scalar("Loss/Entropy Loss 0", entropy_loss_0.cpu().item(), step_cpt)
        writer.add_scalar("Loss/Entropy Loss 1", entropy_loss_1.cpu().item(), step_cpt)

        writer.add_scalar("Loss/Total Loss 0", loss_0.cpu().item(), step_cpt)
        writer.add_scalar("Loss/Total Loss 1", loss_1.cpu().item(), step_cpt)

        print(f"Episode {epoch}, Loss 0 : {loss_0.cpu().item()}, Loss 1 : {loss_1.cpu().item()}")

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