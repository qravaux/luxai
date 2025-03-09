import torch.nn as nn
import torch
import math
import jax.numpy as jnp
import torch.nn.functional as F

class jax_Luxai_Agent(nn.Module) :

    def __init__(self) :

        super(jax_Luxai_Agent,self).__init__()

        self.n_action = 6
        self.n_units = 16
        self.max_sap_range = 8
        self.n_inputs_maps = 10
        self.max_relic_nodes = 6
        self.map_height = 24
        self.map_width = 24

        self.actor_size = [2048,1024,512,256]
        self.cnn_channels = [16,32,64]
        self.cnn_kernels = [9,5,3]
        self.cnn_strides = [1,1,1]
        self.critic_size = [2048,1024,512,256]
        
        self.final_activation = nn.Tanh()
        self.activation = nn.Softplus() #nn.ReLU(), to try ELU, Softplus
        self.gain = math.sqrt(2) # 5/3 For Tanh and sqrt(2) for ReLU

        self.max_pooling = nn.MaxPool2d(kernel_size=2)

        self.cnn_inputs_actor = nn.Conv2d(self.n_inputs_maps,
                                    self.cnn_channels[0],
                                    kernel_size=self.cnn_kernels[0],
                                    padding=self.cnn_kernels[0]-1,
                                    stride=self.cnn_strides[0],
                                    dtype=torch.float32)
        
        self.cnn_hidden_actor = nn.ModuleList([nn.Conv2d(self.cnn_channels[i],
                                     self.cnn_channels[i+1],
                                     kernel_size=self.cnn_kernels[i+1],
                                     padding=self.cnn_kernels[i+1]-1,
                                    stride=self.cnn_strides[i+1],
                                    dtype=torch.float32) for i in range(len(self.cnn_channels)-1)])
        
        self.cnn_inputs_critic = nn.Conv2d(self.n_inputs_maps,
                                    self.cnn_channels[0],
                                    kernel_size=self.cnn_kernels[0],
                                    padding=self.cnn_kernels[0]-1,
                                    stride=self.cnn_strides[0],
                                    dtype=torch.float32)
        
        self.cnn_hidden_critic = nn.ModuleList([nn.Conv2d(self.cnn_channels[i],
                                     self.cnn_channels[i+1],
                                     kernel_size=self.cnn_kernels[i+1],
                                     padding=self.cnn_kernels[i+1]-1,
                                    stride=self.cnn_strides[i+1],
                                    dtype=torch.float32) for i in range(len(self.cnn_channels)-1)])
        
        state_maps = torch.zeros(self.n_inputs_maps,self.map_width,self.map_height)
        state_features = torch.zeros(8*self.n_units + self.max_relic_nodes*3 + 6 + 4)
        
        with torch.no_grad() :
            x = self.cnn_inputs_actor(state_maps)
            x = self.max_pooling(x)
            for layer in self.cnn_hidden_actor :
                x = layer(x)
                x = self.max_pooling(x)
            self.n_inputs_features = x.flatten(start_dim=0).size(0) + state_features.size(0)
        
        self.inputs_actor = nn.Linear(self.n_inputs_features,self.actor_size[0],dtype=torch.float32)
        self.hidden_actor = nn.ModuleList([nn.Linear(self.actor_size[i],self.actor_size[i+1],dtype=torch.float32) for i in range(len(self.actor_size)-1)])

        self.actor_action_layer = nn.Linear(self.actor_size[-1],self.n_action*self.n_units,dtype=torch.float32)
        self.actor_dx_layer = nn.Linear(self.actor_size[-1],(self.max_sap_range*2+1)*self.n_units,dtype=torch.float32)
        self.actor_dy_layer = nn.Linear(self.actor_size[-1],(self.max_sap_range*2+1)*self.n_units,dtype=torch.float32)

        self.inputs_critic = nn.Linear(self.n_inputs_features,self.critic_size[0],dtype=torch.float32)
        self.hidden_critic = nn.ModuleList([nn.Linear(self.critic_size[i],self.critic_size[i+1],dtype=torch.float32) for i in range(len(self.critic_size)-1)])
        self.outputs_critic = nn.Linear(self.critic_size[-1],self.n_units,dtype=torch.float32)

        #Initialization
        nn.init.orthogonal_(self.cnn_inputs_actor.weight,gain=self.gain)
        nn.init.zeros_(self.cnn_inputs_actor.bias)

        for layers in self.cnn_hidden_actor :
            nn.init.orthogonal_(layers.weight,gain=self.gain)
            nn.init.zeros_(layers.bias)

        nn.init.orthogonal_(self.cnn_inputs_critic.weight,gain=self.gain)
        nn.init.zeros_(self.cnn_inputs_critic.bias)

        for layers in self.cnn_hidden_critic :
            nn.init.orthogonal_(layers.weight,gain=self.gain)
            nn.init.zeros_(layers.bias)

        nn.init.orthogonal_(self.inputs_actor.weight,gain=self.gain)
        nn.init.zeros_(self.inputs_actor.bias)

        for layers in self.hidden_actor :
            nn.init.orthogonal_(layers.weight,gain=self.gain)
            nn.init.zeros_(layers.bias)

        nn.init.orthogonal_(self.inputs_critic.weight,gain=self.gain)
        nn.init.zeros_(self.inputs_critic.bias)

        for layers in self.hidden_critic :
            nn.init.orthogonal_(layers.weight,gain=self.gain)
            nn.init.zeros_(layers.bias)

        nn.init.orthogonal_(self.actor_action_layer.weight,gain=0.01)
        nn.init.zeros_(self.actor_action_layer.bias)

        nn.init.orthogonal_(self.actor_dx_layer.weight,gain=0.01)
        nn.init.zeros_(self.actor_dx_layer.bias)

        nn.init.orthogonal_(self.actor_dy_layer.weight,gain=0.01)
        nn.init.zeros_(self.actor_dy_layer.bias)

        nn.init.orthogonal_(self.outputs_critic.weight,gain=1)
        nn.init.zeros_(self.outputs_critic.bias)

    def forward(self,x_maps,x_features,device) :

        x_maps = torch.tensor(x_maps,dtype=torch.float32).to(device)
        x_features = torch.tensor(x_features,dtype=torch.float32).to(device)

        with torch.no_grad() :   
            #Actor part
            x = self.activation(self.cnn_inputs_actor(x_maps))
            x = self.max_pooling(x)
            for layer in self.cnn_hidden_actor :
                x = self.activation(layer(x))
                x = self.max_pooling(x)

            x_input = torch.cat((x.flatten(start_dim=1),x_features),dim=-1)

            x = self.activation(self.inputs_actor(x_input))
            for layer in self.hidden_actor :
                x = self.activation(layer(x))

            actor_action = self.final_activation(self.actor_action_layer(x)).view(-1,self.n_units,self.n_action)
            actor_dx = self.final_activation(self.actor_dx_layer(x)).view(-1,self.n_units,self.max_sap_range*2+1)
            actor_dy = self.final_activation(self.actor_dy_layer(x)).view(-1,self.n_units,self.max_sap_range*2+1)

            #Critic part
            x = self.activation(self.cnn_inputs_critic(x_maps))
            x = self.max_pooling(x)
            for layer in self.cnn_hidden_critic :
                x = self.activation(layer(x))
                x = self.max_pooling(x)

            x_input = torch.cat((x.flatten(start_dim=1),x_features),dim=-1)

            x = self.activation(self.inputs_critic(x_input))
            for layer in self.hidden_critic :
                x = self.activation(layer(x))
            value = self.outputs_critic(x)  

            actor_action = jnp.array(actor_action.cpu(),dtype=jnp.float32)
            actor_dx = jnp.array(actor_dx.cpu(),dtype=jnp.float32)
            actor_dy = jnp.array(actor_dy.cpu(),dtype=jnp.float32)
            value = jnp.array(value.cpu(),dtype=jnp.float32)     

            return actor_action, actor_dx, actor_dy, value
        
    def training_forward(self,x_maps,x_features,action,mask_action,mask_dx,mask_dy) :
        
        #Actor part
        batch_size = x_features.size(0)
        x = self.activation(self.cnn_inputs_actor(x_maps))
        x = self.max_pooling(x)
        for layer in self.cnn_hidden_actor :
            x = self.activation(layer(x))
            x = self.max_pooling(x)
        x_input = torch.cat((x.flatten(start_dim=1),x_features),dim=-1)

        x = self.activation(self.inputs_actor(x_input))
        for layer in self.hidden_actor :
            x = self.activation(layer(x))

        actor_action = self.final_activation(self.actor_action_layer(x)).view(batch_size,self.n_units,self.n_action) - mask_action*100
        actor_dx = self.final_activation(self.actor_dx_layer(x)).view(batch_size,self.n_units,self.max_sap_range*2+1) - mask_dx*100
        actor_dy = self.final_activation(self.actor_dy_layer(x)).view(batch_size,self.n_units,self.max_sap_range*2+1) - mask_dy*100

        actor_action = F.log_softmax(actor_action,dim=-1)
        actor_dx = F.log_softmax(actor_dx,dim=-1)
        actor_dy = F.log_softmax(actor_dy,dim=-1)

        # Computing log probabilities for the actions
        step_indices = torch.arange(batch_size).unsqueeze(1).expand(batch_size, self.n_units)
        unit_indices = torch.arange(self.n_units).unsqueeze(0).expand(batch_size, self.n_units) 

        log_prob = actor_action[step_indices,unit_indices, action[:,:, 0]]
        log_prob += actor_dx[step_indices,unit_indices, action[:,:, 1]+self.max_sap_range]
        log_prob += actor_dy[step_indices,unit_indices, action[:,:, 2]+self.max_sap_range]

        #Critic part
        x = self.activation(self.cnn_inputs_critic(x_maps))
        x = self.max_pooling(x)
        for layer in self.cnn_hidden_critic :
            x = self.activation(layer(x))
            x = self.max_pooling(x)
        x_input = torch.cat((x.flatten(start_dim=1),x_features),dim=-1)

        x = self.activation(self.inputs_critic(x_input))
        for layer in self.hidden_critic :
            x = self.activation(layer(x))
        value = self.outputs_critic(x)

        return value,log_prob
    