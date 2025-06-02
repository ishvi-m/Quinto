import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from commons.quartoenv.env_v2 import RandomOpponentEnv_V2

# --- Q-Network ---
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, state):
        return self.net(state)

# --- Replay Buffer (for offline dataset) ---
class ReplayBuffer:
    def __init__(self, states, actions, rewards, next_states, dones):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones
        self.size = len(states)
    def sample(self, batch_size):
        idx = np.random.choice(self.size, batch_size)
        return (self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx], self.dones[idx])

# --- CQL Agent ---
class CQLAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=1e-3, gamma=0.99, alpha=1.0):
        self.q_net = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_net = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.alpha = alpha
        self.action_dim = action_dim
    def update(self, batch):
        s, a, r, s_next, d = batch
        q_values = self.q_net(s)
        q_sa = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_q_net(s_next)
            max_next_q = next_q_values.max(1)[0]
            target = r + self.gamma * (1 - d) * max_next_q
        bellman_loss = ((q_sa - target) ** 2).mean()
        logsumexp_q = torch.logsumexp(q_values, dim=1).mean()
        data_q = q_sa.mean()
        cql_penalty = logsumexp_q - data_q
        loss = bellman_loss + self.alpha * cql_penalty
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), bellman_loss.item(), cql_penalty.item()
    def update_target(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())
    def select_action(self, state):
        with torch.no_grad():
            q_values = self.q_net(state)
            action_idx = q_values.argmax().item()
        return action_idx

# --- Data Loading and Preparation ---
with open("offline_quinto_dataset.pkl", "rb") as f:
    data = pickle.load(f)

states = np.stack([d['state'] for d in data])
actions = np.stack([d['action'] for d in data])
rewards = np.array([d['reward'] for d in data], dtype=np.float32)
next_states = np.stack([d['next_state'] for d in data])
dones = np.array([d['done'] for d in data], dtype=np.float32)

# Encode actions as single integer: pos * 16 + piece
actions = np.array([a[0] * 16 + a[1] for a in actions])

# Convert to torch tensors
states = torch.tensor(states, dtype=torch.float32)
actions = torch.tensor(actions, dtype=torch.long)
rewards = torch.tensor(rewards, dtype=torch.float32)
next_states = torch.tensor(next_states, dtype=torch.float32)
dones = torch.tensor(dones, dtype=torch.float32)

state_dim = states.shape[1]
action_dim = 16 * 16

buffer = ReplayBuffer(states, actions, rewards, next_states, dones)
agent = CQLAgent(state_dim, action_dim)

# --- Evaluation Function ---
def evaluate_policy(agent, num_episodes=20):
    env = RandomOpponentEnv_V2()
    wins = 0
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action_idx = agent.select_action(state)
            pos = action_idx // 16
            piece = action_idx % 16
            action = (pos, piece)
            obs, reward, done, info = env.step(action)
        if info.get('win', False):
            wins += 1
    return wins / num_episodes

# --- Training Loop ---
batch_size = 64
num_steps = 10000
writer = SummaryWriter(log_dir="runs/offline_cql_cs224r")

for step in range(num_steps):
    batch = buffer.sample(batch_size)
    loss, bellman_loss, cql_penalty = agent.update(batch)
    writer.add_scalar('Loss/total', loss, step)
    writer.add_scalar('Loss/bellman', bellman_loss, step)
    writer.add_scalar('Loss/cql_penalty', cql_penalty, step)
    if step % 100 == 0:
        agent.update_target()
        print(f"Step {step}, Loss: {loss:.4f}, Bellman: {bellman_loss:.4f}, CQL: {cql_penalty:.4f}")
    if step % 500 == 0:
        win_rate = evaluate_policy(agent, num_episodes=20)
        writer.add_scalar('Eval/win_rate', win_rate, step)
        print(f"Step {step}, Win Rate: {win_rate:.2f}")

writer.close()



