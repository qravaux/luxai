import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
from src.luxai_s3.wrappers import LuxAIS3GymEnv
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from policies import Luxai_Agent
from workers import Luxai_Worker

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
    lr0 = 1e-5
    lr1 = 1e-5
    max_norm0 = 0.5
    max_norm1 = 0.5
    entropy_coef0 = 0.01
    entropy_coef1 = 0.01
    weight_decay_0 = 0
    weight_decay_1 = 0


    batch_size = 100
    vf_coef = 1
    gamma = 1
    gae_lambda = 0.99
    save_rate = 20

    n_epochs = 10#int(1e6)
    n_batch = 10
    num_workers = 6
    n_episode = 4
    n_steps = 100

    file_name = 'experiment_13'
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

    model_0 = Luxai_Agent('player_0')
    model_0.share_memory()  # For multiprocessing, the model parameters must be shared
    optimizer_0 = torch.optim.Adam(model_0.parameters(), lr=lr0, weight_decay=weight_decay_0)

    model_1 = Luxai_Agent('player_1')
    model_1.share_memory()  # For multiprocessing, the model parameters must be shared
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=lr1, weight_decay=weight_decay_1)

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

        states_maps_0.requires_grad = False
        states_features_0.requires_grad = False
        advantages_0.requires_grad = False
        returns_0.requires_grad = False

        states_maps_1 = torch.cat(states_maps_1,dim=0)
        states_features_1 = torch.cat(states_features_1,dim=0)
        actions_1 = torch.cat(actions_1,dim=0)
        advantages_1 = torch.cat(advantages_1,dim=0)
        returns_1 = torch.cat(returns_1,dim=0)
        mask_action_1 = torch.cat(mask_action_1,dim=0)
        mask_dx_1 = torch.cat(mask_dx_1,dim=0)
        mask_dy_1 = torch.cat(mask_dy_1,dim=0)

        states_maps_1.requires_grad = False
        states_features_1.requires_grad = False
        advantages_1.requires_grad = False
        returns_1.requires_grad = False

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