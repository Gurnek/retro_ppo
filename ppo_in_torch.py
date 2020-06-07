import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optimizer
from tqdm import tqdm

class Actor(nn.Module):
    def __init__(self, in_channels, out_space):
        super(Actor, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, 8, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 32, 5, stride=1)
        self.lin1 = nn.Linear(7*7*32, 1024)
        self.lin2 = nn.Linear(1024, 512)
        self.lin3 = nn.Linear(512, out_space)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.max_pool(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.softmax(self.lin3(x), dim=1)
        return x

class Critic(nn.Module):
    def __init__(self, in_channels):
        super(Critic, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, 8, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 32, 5, stride=1)
        self.lin1 = nn.Linear(7*7*32, 1024)
        self.lin2 = nn.Linear(1024, 512)
        self.lin3 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.max_pool(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x
    
class Memory:
    def __init__(self):
        self.states_ = []
        self.actions_ = []
        self.rewards_ = []
        self.terminals_ = []
        self.probs_ = []

    def reset(self):
        del self.states_[:]
        del self.actions_[:]
        del self.rewards_[:]
        del self.terminals_[:]
        del self.probs_[:]
        self.clear()

    def clear(self):
        self.states_ = []
        self.actions_ = []
        self.rewards_ = []
        self.terminals_ = []
        self.probs_ = []

class PPO:
    def __init__(self, in_channels, out_space, gamma, lr, device='cuda'):
        self.actor = Actor(in_channels, out_space).to(device)
        self.critic = Critic(in_channels).to(device)
        self.old_actor = Actor(in_channels, out_space).to(device)
        self.old_critic = Critic(in_channels).to(device)

        self.old_actor.load_state_dict(self.actor.state_dict())
        self.old_critic.load_state_dict(self.critic.state_dict())

        self.opt = optimizer.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

        self.memory = Memory()

        self.gamma = gamma
        self.device = device

    def update(self):
        # Calculating the losses
        q_vals = []
        last = 0
        for reward, terminal in zip(reversed(self.memory.rewards_), reversed(self.memory.terminals_)):
            if terminal:
                q_vals.insert(0, reward)
                last = 0
            else:
                q_vals.insert(0, last*self.gamma + reward)
                last = last*self.gamma + reward
        q_vals = torch.tensor(q_vals).to(self.device)
        
        states = torch.stack(self.memory.states_).view(-1, 4, 100, 100).to(self.device)
        actions = torch.stack(self.memory.actions_).to(self.device)
        probs = torch.stack(self.memory.probs_).to(self.device)

        #Update for K=5 epochs
        for _ in range(5):
            act_probs = self.actor(states)
            dist = Categorical(act_probs)
            log_probs = dist.log_prob(actions)

            values = self.critic(states)

            ratios = torch.exp(log_probs - probs)
            advantages = q_vals - values

            lclip = torch.min(ratios * advantages, torch.clamp(ratios, 1.2, 0.8) * advantages)
            lvf = F.mse_loss(values.squeeze(), q_vals)
            ls = dist.entropy()
            loss = -lclip + 0.6 * lvf - 0.05*ls

            self.opt.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.opt.step()

        self.old_actor.load_state_dict(self.actor.state_dict())
        self.old_critic.load_state_dict(self.critic.state_dict())


from env import DoomEnv
e = DoomEnv('./scenarios/deadly_corridor.cfg')
ppo = PPO(4, e.action_space(), 0.99, 0.003)

timestep = 0

for i in tqdm(range(1000)):
    state = e.reset()
    while not e.done():
        timestep += 1

        act_probs = ppo.old_actor(state)
        dist = Categorical(act_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        reward = e.action(action.item())
        new_state = e.get_state()

        ppo.memory.states_.append(state)
        ppo.memory.actions_.append(action)
        ppo.memory.probs_.append(log_prob)
        ppo.memory.rewards_.append(reward)
        ppo.memory.terminals_.append(e.done())
        state = new_state

        if timestep % 100 == 0:
            ppo.update()
            ppo.memory.reset()
            timestep = 0

    print(e.game_.get_total_reward())