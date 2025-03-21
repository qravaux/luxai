import os
from jax_utils import *
from jax_policies import jax_Luxai_Agent
from jax_worker import jax_Luxai_Worker
import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":

    if not os.path.exists("runs"):
        # Create directory
        os.makedirs("runs")

    if not os.path.exists("policy"):
        # Create directory
        os.makedirs("policy")

    print('Initialise training environment...\n')
    num_envs = 333
    num_workers = 3

    lr = 1e-3
    lr_decay = 0.999
    max_norm = 0.5
    entropy_coef = 0.001
    weight_decay = 0.001
    eps = 1e-8
    betas = (0.9,0.999)
    lmbda = lambda epoch: lr_decay

    replay_buffer_capacity = 101000
    batch_size = 1024
    vf_coef = 0.5
    gamma = 0.995
    gae_lambda = 0.99
    save_rate = 10

    n_epochs = int(1e9)
    n_steps = 101
    n_batch = ((num_envs*n_steps) // batch_size)

    file_name = 'test_1_episode'
    save_dir = f"policy/{file_name}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(f"runs/{file_name}")

    #Create the policy and Replay Buffer
    policy = jax_Luxai_Agent()
    policy_gpu = jax_Luxai_Agent()
    policy_gpu.load_state_dict(policy.state_dict())
    policy_gpu.to(device)
    policy.share_memory()
    queue = mp.Queue(maxsize=num_workers)
    reward_queue = mp.Queue(maxsize=num_workers)
    point_queue = mp.Queue(maxsize=num_workers)
    intermediate_reward = mp.Queue(maxsize=num_workers)

    optimizer = torch.optim.Adam(policy_gpu.parameters(), lr=lr, weight_decay=weight_decay, eps=eps, betas=betas)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer,lmbda)

    buffer = PrioritizedReplayBuffer(replay_buffer_capacity,device)
    
    if not os.path.exists(save_dir):
        # Créer le dossier
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"policy_step_00.pth")
    torch.save(policy.state_dict(), save_path)

    print('Instantiate workers...')         
    workers = []
    for i in range(num_workers) :
        worker = jax_Luxai_Worker(i,
                                policy,
                                queue,
                                reward_queue,
                                point_queue,
                                intermediate_reward,
                                num_envs,
                                num_workers,
                                gamma, 
                                gae_lambda, 
                                n_steps,
                                n_epochs,
                                file_name,
                                'cpu')
        print(f'-------------------------worker {i} is ready')
        workers.append(worker)
        worker.start()

    print('Done\n')

    while queue.qsize() < num_workers :
            waiting = True
    # Main training loop: Collect experiences from worker and update the model
    print('Start training...\n')

    reward_cpt = 0
    int_rew_cpt = 0

    for epoch in range(n_epochs):

        # Collect data from worker
        while queue.qsize() > 0 :
            states_maps,states_features,actions,advantages,returns,log_probs,mask_actions,mask_dxs,mask_dys,units_mask = queue.get()

            buffer.add(states_maps,
                        states_features,
                        actions,
                        advantages,
                        returns,
                        log_probs,
                        mask_actions,
                        mask_dxs,
                        mask_dys,
                        units_mask)
            
        while reward_queue.qsize() > 0 :
            reward = reward_queue.get()
            point = point_queue.get()
            for ev in range(num_envs) :
                reward_cpt += 505

                writer.add_scalar("Reward 0", reward[0,ev].item(), reward_cpt)
                writer.add_scalar("Reward 1", reward[1,ev].item(), reward_cpt)

                writer.add_scalar("Point 0", point[0,ev].item(), reward_cpt)
                writer.add_scalar("Point 1", point[1,ev].item(), reward_cpt)

        while intermediate_reward.qsize() > 0 :
            int_rew = intermediate_reward.get()
            for ev in range(num_envs) :
                int_rew_cpt += n_steps
                writer.add_scalar("Intermediate reward",int_rew[ev].item(), int_rew_cpt)
               

        #Train policy
        for _ in range(n_batch) :

            states_maps_, states_features_, actions_, advantages_, returns_, p_log_probs_, mask_actions_, mask_dxs_, mask_dys_,units_mask_, weights_, indices_ = buffer.sample(batch_size=batch_size,device=device)

            #Compute log_probs and values
            values_,log_probs_ = policy_gpu.training_forward(states_maps_,states_features_,actions_,mask_actions_,mask_dxs_,mask_dys_)
            #advantages_ = (advantages_ - torch.mean(advantages_,dim=0)) / (torch.std(advantages_,dim=0) + 1e-8)
            #returns_ = (returns_ - torch.mean(returns_,dim=0)) / (torch.std(returns_,dim=0) + 1e-8)
        
            # Losses
            entropy_loss = -torch.mean(weights_*torch.mean(units_mask_ * torch.exp(log_probs_) * log_probs_,dim=-1))
            policy_loss = -torch.mean(weights_*torch.mean(units_mask_ * log_probs_ * advantages_,dim=-1))
            value_loss = torch.mean(weights_*torch.mean(units_mask_ * torch.pow(values_ - returns_,2),dim=-1))

            loss = policy_loss + vf_coef *value_loss + entropy_coef * entropy_loss

            # Update model
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=max_norm)
            optimizer.step()

            buffer.update_priorities(batch_indices=indices_,batch_priorities=torch.clamp(torch.sum(advantages_,dim=-1),min=1e-8).detach())

        scheduler.step()
        policy.load_state_dict(policy_gpu.state_dict())

        writer.add_scalar("Loss/Policy Loss", policy_loss.cpu().item(), epoch)
        writer.add_scalar("Loss/Value Loss", value_loss.cpu().item(), epoch)
        writer.add_scalar("Loss/Entropy Loss", entropy_loss.cpu().item(), epoch)
        writer.add_scalar("Loss/Total Loss", loss.cpu().item(), epoch)

        if epoch % save_rate == 0 :
            if not os.path.exists(save_dir):
                # Create directory
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, f"policy_step_{epoch}.pth")
            torch.save(policy.state_dict(), save_path)

    for worker in workers :
        worker.join()
    
    writer.close()

