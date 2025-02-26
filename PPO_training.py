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

class ReplayBuffer(Dataset):
    def __init__(self,states_maps,states_features,actions,advantages,returns,mask_action,mask_dx,mask_dy,log_probs,device):

        self.states_maps = states_maps.to(device)
        self.states_features = states_features.to(device)
        self.actions = actions.to(device)
        self.advantages = advantages.to(device)
        self.returns = returns.to(device)
        self.mask_action = mask_action.to(device)
        self.mask_dx = mask_dx.to(device)
        self.mask_dy = mask_dy.to(device)
        self.log_probs = log_probs.to(device)

    def __len__(self):
        return len(self.states_maps)

    def __getitem__(self, idx):
        return self.states_maps[idx],self.states_features[idx],self.actions[idx],self.advantages[idx],self.returns[idx],self.mask_action[idx],self.mask_dx[idx],self.mask_dy[idx],self.log_probs[idx]
                
if __name__ == "__main__":

    print('Initialise training environment...\n')
    lr0 = 1e-5
    lr1 = 5e-6
    max_norm0 = 0.5
    max_norm1 = 0.5
    entropy_coef0 = 0.05
    entropy_coef1 = 0.05
    weight_decay_0 = 1e-3
    weight_decay_1 = 1e-3

    clip_coef = 0.2
    vf_coef = 1
    gamma = 1
    gae_lambda = 0.99

    eps = 1e-8
    betas = (0.9,0.999)

    batch_size = 100
    save_rate = 100
    n_epochs = int(1e6)
    n_batch = 10
    num_workers = 6
    n_episode = 5
    n_steps = 100

    file_name = 'PPO_unit_reward'
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

    shared_queue = mp.Queue()  # Queue to share data between workers and the main process
    reward_queue = mp.Queue()
    point_queue = mp.Queue()
    event = mp.Event()

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
    
    # Main training loop: Collect experiences from workers and update the model
    print('Start training...\n')
    for epoch in range(n_epochs):

        # Collect data from workers
        experiences_0 = []
        experiences_1 = []

        for _ in range(n_episode):
            for _ in range(num_workers) :

                states_maps,states_features,actions,advantages,returns,mask_actions,mask_dxs,mask_dys,log_probs = shared_queue.get()

                experiences_0.append((states_maps[0],states_features[0],actions[0],advantages[0],returns[0],mask_actions[0],mask_dxs[0],mask_dys[0],log_probs[0]))
                experiences_1.append((states_maps[1],states_features[1],actions[1],advantages[1],returns[1],mask_actions[1],mask_dxs[1],mask_dys[1],log_probs[1]))

        event.set()

        # Process the collected experiences
        states_maps_0,states_features_0,actions_0,advantages_0,returns_0,mask_action_0,mask_dx_0,mask_dy_0,log_probs_0 = zip(*experiences_0)
        states_maps_1,states_features_1,actions_1,advantages_1,returns_1,mask_action_1,mask_dx_1,mask_dy_1,log_probs_1 = zip(*experiences_1)

        states_maps_0 = torch.cat(states_maps_0,dim=0)
        states_features_0 = torch.cat(states_features_0,dim=0)
        actions_0 = torch.cat(actions_0,dim=0)
        advantages_0 = torch.cat(advantages_0,dim=0)
        returns_0 = torch.cat(returns_0,dim=0)
        mask_action_0 = torch.cat(mask_action_0,dim=0)
        mask_dx_0 = torch.cat(mask_dx_0,dim=0)
        mask_dy_0 = torch.cat(mask_dy_0,dim=0)
        log_probs_0 = torch.cat(log_probs_0,dim=0)

        states_maps_0.requires_grad = False
        states_features_0.requires_grad = False
        advantages_0.requires_grad = False
        returns_0.requires_grad = False
        log_probs_0.requires_grad = False

        states_maps_1 = torch.cat(states_maps_1,dim=0)
        states_features_1 = torch.cat(states_features_1,dim=0)
        actions_1 = torch.cat(actions_1,dim=0)
        advantages_1 = torch.cat(advantages_1,dim=0)
        returns_1 = torch.cat(returns_1,dim=0)
        mask_action_1 = torch.cat(mask_action_1,dim=0)
        mask_dx_1 = torch.cat(mask_dx_1,dim=0)
        mask_dy_1 = torch.cat(mask_dy_1,dim=0)
        log_probs_1 = torch.cat(log_probs_1,dim=0)

        states_maps_1.requires_grad = False
        states_features_1.requires_grad = False
        advantages_1.requires_grad = False
        returns_1.requires_grad = False
        log_probs_1.requires_grad = False

        train_data_0 = ReplayBuffer(states_maps_0,states_features_0,actions_0,advantages_0,returns_0,mask_action_0,mask_dx_0,mask_dy_0,log_probs_0,device)
        train_data_1 = ReplayBuffer(states_maps_1,states_features_1,actions_1,advantages_1,returns_1,mask_action_1,mask_dx_1,mask_dy_1,log_probs_1,device)
 
        train_loader_0 = DataLoader(train_data_0, batch_size=batch_size, shuffle=True)
        train_loader_1 = DataLoader(train_data_1, batch_size=batch_size, shuffle=True)

        for _ in range(n_batch) :

            for batch in train_loader_0 :

                states_maps_,states_features_,actions_,advantages_,returns_,mask_action_,mask_dx_,mask_dy_,p_log_probs_ = batch

                #Compute log_probs and values
                values_,log_probs_ = model_0_gpu.training_forward(states_maps_,states_features_,actions_,mask_action_,mask_dx_,mask_dy_)

                advantages_ = (advantages_ - torch.mean(advantages_,dim=0)) / (torch.std(advantages_,dim=0) + 1e-8)

                # Losses
                ratio = torch.exp(log_probs_-p_log_probs_)
                policy_loss_0 = -torch.mean(torch.min(advantages_ * ratio, advantages_ * torch.clamp(ratio,1-clip_coef,1+clip_coef)))
                entropy_loss_0 = -torch.mean(torch.exp(log_probs_) * log_probs_)
                value_loss_0 = torch.mean(torch.pow(values_ - returns_,2))

                loss_0 = policy_loss_0 + vf_coef *value_loss_0 + entropy_coef0 * entropy_loss_0

                # Update model
                optimizer_0.zero_grad()
                loss_0.backward()
                torch.nn.utils.clip_grad_norm_(model_0_gpu.parameters(), max_norm=max_norm0)
                optimizer_0.step()

            for batch in train_loader_1 :

                states_maps_,states_features_,actions_,advantages_,returns_,mask_action_,mask_dx_,mask_dy_,p_log_probs_ = batch

                #Compute log_probs and values
                values_,log_probs_ = model_1_gpu.training_forward(states_maps_,states_features_,actions_,mask_action_,mask_dx_,mask_dy_)

                advantages_ = (advantages_ - torch.mean(advantages_,dim=0)) / (torch.std(advantages_,dim=0) + 1e-8)

                # Losses
                ratio = torch.exp(log_probs_-p_log_probs_)
                policy_loss_1 = -torch.mean(torch.min(advantages_ * ratio, advantages_ * torch.clamp(ratio,1-clip_coef,1+clip_coef)))
                entropy_loss_1 = -torch.mean(torch.exp(log_probs_) * log_probs_)  
                value_loss_1 = torch.mean(torch.pow(values_ - returns_,2))

                loss_1 = policy_loss_1 + vf_coef *value_loss_1 + entropy_coef1 * entropy_loss_1

                # Update model
                optimizer_1.zero_grad()
                loss_1.backward()
                torch.nn.utils.clip_grad_norm_(model_1_gpu.parameters(), max_norm=max_norm1)
                optimizer_1.step()


        model_0.load_state_dict(model_0_gpu.state_dict())
        model_1.load_state_dict(model_1_gpu.state_dict())

        try :
            while True :
                reward = reward_queue.get_nowait()
                point = point_queue.get_nowait()
                reward_cpt += len_episode

                writer.add_scalar("Reward 0", reward[0].item(), reward_cpt)
                writer.add_scalar("Reward 1", reward[1].item(), reward_cpt)

                writer.add_scalar("Point 0", point[0].item(), reward_cpt)
                writer.add_scalar("Point 1", point[1].item(), reward_cpt)

        except :
            None

        step_cpt += n_episode*n_steps*num_workers

        writer.add_scalar("Loss/Policy Loss 0", policy_loss_0.cpu().item(), step_cpt)
        writer.add_scalar("Loss/Policy Loss 1", policy_loss_1.cpu().item(), step_cpt)

        writer.add_scalar("Loss/Value Loss 0", value_loss_0.cpu().item(), step_cpt)
        writer.add_scalar("Loss/Value Loss 1", value_loss_1.cpu().item(), step_cpt)

        writer.add_scalar("Loss/Entropy Loss 0", entropy_loss_0.cpu().item(), step_cpt)
        writer.add_scalar("Loss/Entropy Loss 1", entropy_loss_1.cpu().item(), step_cpt)

        writer.add_scalar("Loss/Total Loss 0", loss_0.cpu().item(), step_cpt)
        writer.add_scalar("Loss/Total Loss 1", loss_1.cpu().item(), step_cpt)

        print(f"Episode {epoch}, Loss 0 : {loss_0.cpu().item()}, Loss 1 : {loss_1.cpu().item()}")

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