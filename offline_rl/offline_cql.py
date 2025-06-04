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
from commons.quartoenv.env_v4 import CustomOpponentEnv_V4
import argparse


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

class ReplayBuffer:
    def __init__(self, states, actions, rewards, next_states, dones, legal_actions, next_legal_actions):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones
        self.legal_actions = legal_actions
        self.next_legal_actions = next_legal_actions
        self.size = len(states)
    def sample(self, batch_size):
        idx = np.random.choice(self.size, batch_size)
        return (self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx], self.dones[idx],
                [self.legal_actions[i] for i in idx], [self.next_legal_actions[i] for i in idx])

class CQLAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=1e-3, gamma=0.99, alpha=1.0):
        self.q_net = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_net = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.alpha = alpha
        self.action_dim = action_dim

    def update(self, batch, legal_actions_batch, next_legal_actions_batch):
        s, a, r, s_next, d = batch
        q_values = self.q_net(s)
        q_sa = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_q_net(s_next)
            max_next_q = torch.stack([
                next_q_values[i][next_legal_actions_batch[i]].max()
                for i in range(len(next_legal_actions_batch))
            ])
            target = r + self.gamma * (1 - d) * max_next_q
        # CQL penalty: use only legal actions for each sample
        logsumexp_q = torch.stack([
            torch.logsumexp(q_values[i][legal_actions_batch[i]], dim=0)
            for i in range(len(legal_actions_batch))
        ]).mean()
        data_q = q_sa.mean()
        cql_penalty = logsumexp_q - data_q
        bellman_loss = ((q_sa - target) ** 2).mean()
        loss = bellman_loss + self.alpha * cql_penalty
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), bellman_loss.item(), cql_penalty.item()
    def update_target(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())
    def select_action(self, state, legal_actions=None):
        with torch.no_grad():
            q_values = self.q_net(state)
            if legal_actions is not None:
                # Mask out illegal actions
                mask = torch.zeros_like(q_values)
                mask[0, legal_actions] = 1
                q_values = q_values * mask + (1 - mask) * -1e9  # Large negative for illegal
            action_idx = q_values.argmax().item()
        return action_idx

with open("offline_quinto_dataset.pkl", "rb") as f:
    data = pickle.load(f)

states = np.stack([d['state'] for d in data])
actions = np.stack([d['action'] for d in data])
rewards = np.array([d['reward'] for d in data], dtype=np.float32)
next_states = np.stack([d['next_state'] for d in data])
dones = np.array([d['done'] for d in data], dtype=np.float32)

# actions as single integer: pos * 16 + piece
actions = np.array([a[0] * 16 + a[1] for a in actions])

# convert to torch tensors
states = torch.tensor(states, dtype=torch.float32)
actions = torch.tensor(actions, dtype=torch.long)
rewards = torch.tensor(rewards, dtype=torch.float32)
next_states = torch.tensor(next_states, dtype=torch.float32)
dones = torch.tensor(dones, dtype=torch.float32)

state_dim = states.shape[1]
action_dim = 16 * 16

buffer = ReplayBuffer(states, actions, rewards, next_states, dones, [d['legal_actions'] for d in data], [d['next_legal_actions'] for d in data])
agent = CQLAgent(state_dim, action_dim)

def evaluate_policy(agent, num_episodes=20):
    env = CustomOpponentEnv_V4()
    wins = 0
    losses = 0
    draws = 0
    invalids = 0
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            legal_actions = [pos * 16 + piece for (pos, piece) in env.legal_actions() if piece is not None]
            action_idx = agent.select_action(state, legal_actions=legal_actions)
            pos = action_idx // 16
            piece = action_idx % 16
            action = (pos, piece)
            obs, reward, done, _, info = env.step(action)
            next_legal_actions = [pos * 16 + piece for (pos, piece) in env.legal_actions() if piece is not None]
            data = {
                "state": obs,
                "action": action,
                "reward": reward,
                "next_state": obs,
                "done": done,
                "info": info,
                "legal_actions": legal_actions,
                "next_legal_actions": next_legal_actions
            }
            # ... append data to dataset ...
        if info.get('win', False):
            wins += 1
        elif info.get('draw', False):
            draws += 1
        elif info.get('invalid', False):
            invalids += 1
        else:
            losses += 1
    total = num_episodes
    return {
        "win": wins / total * 100,
        "loss": losses / total * 100,
        "draw": draws / total * 100,
        "invalid": invalids / total * 100
    }

# training
batch_size = 64
num_steps = 100000
parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, default='runs/offline_cql_cs224r', help='TensorBoard log directory')
parser.add_argument('--env_version', type=str, default='v4', choices=['v2', 'v4'], help='Environment version to use for evaluation')
args = parser.parse_args()
writer = SummaryWriter(log_dir=args.logdir)

for step in range(num_steps):
    batch = buffer.sample(batch_size)
    loss, bellman_loss, cql_penalty = agent.update(batch[:5], batch[5], batch[6])
    writer.add_scalar('Loss/total', loss, step)
    writer.add_scalar('Loss/bellman', bellman_loss, step)
    writer.add_scalar('Loss/cql_penalty', cql_penalty, step)
    if step % 100 == 0:
        agent.update_target()
        print(f"Step {step}, Loss: {loss:.4f}, Bellman: {bellman_loss:.4f}, CQL: {cql_penalty:.4f}")
    if step % 500 == 0:
        eval_stats = evaluate_policy(agent, num_episodes=100)
        writer.add_scalar('Eval/win_pct', eval_stats["win"], step)
        writer.add_scalar('Eval/loss_pct', eval_stats["loss"], step)
        writer.add_scalar('Eval/draw_pct', eval_stats["draw"], step)
        writer.add_scalar('Eval/invalid_pct', eval_stats["invalid"], step)
        print(f"Step {step}, Win: {eval_stats['win']:.2f}%, Loss: {eval_stats['loss']:.2f}%, Draw: {eval_stats['draw']:.2f}%, Invalid: {eval_stats['invalid']:.2f}%")

writer.close()



