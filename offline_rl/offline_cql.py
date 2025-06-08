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
            max_next_q = []
            for i in range(len(next_legal_actions_batch)):
                indices = [x for x in next_legal_actions_batch[i] if isinstance(x, int)]
                if d[i] == 1.0:
                    max_next_q.append(torch.tensor(0.0, device=next_q_values.device))  # Terminal: no bootstrapping
                elif len(indices) > 0:
                    max_next_q.append(next_q_values[i][indices].max())
                else:
                    max_next_q.append(torch.tensor(0.0, device=next_q_values.device))  # No legal actions: treat as terminal
            max_next_q = torch.stack(max_next_q)
            target = r + self.gamma * (1 - d) * max_next_q
        # CQL penalty: use only legal actions for each sample
        logsumexp_values = []
        for i in range(len(legal_actions_batch)):
            indices = [x for x in legal_actions_batch[i] if isinstance(x, int)]
            if len(indices) == 0:
                continue
            try:
                logsumexp_values.append(torch.logsumexp(q_values[i][indices], dim=0))
            except Exception as e:
                print(f"Skipping sample {i} due to indexing error: {e} | indices: {legal_actions_batch[i]}")
                continue
        if len(logsumexp_values) > 0:
            logsumexp_q = torch.stack(logsumexp_values).mean()
        else:
            logsumexp_q = torch.tensor(0.0, device=q_values.device)
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
            q_legal = q_values[0, legal_actions]
            best_idx = int(torch.argmax(q_legal).item())
            best_action = legal_actions[best_idx]
        return best_action

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='offline_quinto_dataset.pkl', help='Path to offline dataset (.pkl)')
parser.add_argument('--logdir', type=str, default='runs/offline_cql_cs224r', help='TensorBoard log directory')
parser.add_argument('--env_version', type=str, default=None, help='Environment version to use for evaluation. If omitted, will use value stored in dataset.')
parser.add_argument('--normalize_rewards', action='store_true', help='Normalize rewards using mean/std of dataset')
args, _ = parser.parse_known_args()

# Load dataset
with open(args.dataset, "rb") as f:
    data = pickle.load(f)

# Determine env version from dataset if not explicitly provided
if args.env_version is None:
    dataset_env_version = data[0].get('env_version', 'v4')
    args.env_version = dataset_env_version

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

# Optionally normalize rewards to have zero mean and unit std (helps across env versions with different scales)
if args.normalize_rewards:
    mean_r = rewards.mean()
    std_r = rewards.std()
    if std_r > 1e-8:
        rewards = (rewards - mean_r) / std_r
    else:
        rewards = rewards * 0.0  # all same
    rewards = torch.tensor(rewards, dtype=torch.float32)
else:
    rewards = torch.tensor(rewards, dtype=torch.float32)

# Ensure next_states and dones tensors created after potential reward normalization logic
next_states = torch.tensor(next_states, dtype=torch.float32)
dones = torch.tensor(dones, dtype=torch.float32)

state_dim = states.shape[1]
action_dim = 16 * 16

buffer = ReplayBuffer(states, actions, rewards, next_states, dones, [d['legal_actions'] for d in data], [d['next_legal_actions'] for d in data])
agent = CQLAgent(state_dim, action_dim)

# Create TensorBoard writer
writer = SummaryWriter(log_dir=args.logdir)

# ---------------------------------------------------------------------------
# Helper utilities for evaluation (env construction, obs flattening, legal actions)
# ---------------------------------------------------------------------------

def make_eval_env(version: str):
    if version == 'v2':
        return RandomOpponentEnv_V2()
    elif version == 'v4':
        return CustomOpponentEnv_V4()
    else:
        raise ValueError(f"Unknown env_version '{version}' for evaluation.")


def flatten_obs(obs):
    """Convert (board, piece) or already-flat obs into 1-D float32 numpy array."""
    if isinstance(obs, tuple):
        board, piece = obs
        board_flat = board.flatten().astype(np.float32)
        if piece is None:
            piece_val = -1.0
        elif hasattr(piece, 'index'):
            piece_val = float(piece.index)
        else:
            piece_val = float(piece)
        return np.concatenate([board_flat, np.array([piece_val], dtype=np.float32)])
    else:
        return np.array(obs, dtype=np.float32)


def get_legal_actions(env):
    """Return list of integer-encoded legal actions for the current env state."""
    return list(env.legal_actions())

def evaluate_policy(agent, num_episodes=20, log_episode:bool=False, save_path:str=None, save_board_every_step:bool=False):
    env = make_eval_env(args.env_version)
    wins = 0
    losses = 0
    draws = 0
    invalids = 0
    total_rewards = 0.0
    total_bad_piece = 0
    total_threat_created = 0
    total_threat_blocked = 0
    total_turns = 0
    episode_boards = []

    for ep in range(num_episodes):
        raw_obs = env.reset()
        if isinstance(raw_obs, tuple):
            obs_vec = flatten_obs(raw_obs[0] if len(raw_obs) == 2 else raw_obs)
        else:
            obs_vec = flatten_obs(raw_obs)
        done = False
        step_counter = 0
        while not done:
            state = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
            legal_actions = [(int(row), int(piece)) for (row, piece) in env.legal_actions() if piece is not None]
            legal_action_indices = [row*16 + piece for (row, piece) in legal_actions]
            q_values = agent.q_net(state)
            if len(legal_actions) == 0:
                # If no legal actions but not done, forcibly end episode (silent)
                done = True
                break
            elif len(legal_action_indices) == 0:
                import random
                best_action = random.choice(legal_actions)
            else:
                q_legal = q_values[0, legal_action_indices]
                best_idx = int(torch.argmax(q_legal).item())
                best_action = legal_actions[best_idx]
            step_result = env.step(best_action)
            # Print info after step
            if len(step_result) == 5:
                raw_obs, reward, done, truncated, info = step_result
            else:
                raw_obs, reward, done, info = step_result
            obs_vec = flatten_obs(raw_obs)
            total_rewards += reward
            total_bad_piece += int(info.get('bad_piece', False))
            total_threat_created += int(info.get('threat_created', False))
            total_threat_blocked += int(info.get('threat_blocked', False))
            step_counter += 1
            if log_episode and ep == 0 and save_board_every_step:
                episode_boards.append(np.copy(env.game.board))
            # If done after step, break immediately
            if done:
                break
        if info.get('win', False):
            wins += 1
        elif info.get('draw', False):
            draws += 1
        elif info.get('invalid', False):
            invalids += 1
        else:
            losses += 1
        total_turns += step_counter

    total = num_episodes
    stats = {
        "win": wins / total * 100,
        "loss": losses / total * 100,
        "draw": draws / total * 100,
        "invalid": invalids / total * 100,
        "avg_reward": total_rewards / total if total > 0 else 0.0,
        "avg_bad_piece": total_bad_piece / total,
        "avg_threats_created": total_threat_created / total,
        "avg_threats_blocked": total_threat_blocked / total,
        "avg_turns": total_turns / total if total > 0 else 0.0
    }

    # Save episode boards if requested
    if log_episode and save_path is not None and episode_boards:
        with open(save_path, "wb") as f:
            pickle.dump(episode_boards, f)

    return stats

# training
batch_size = 64
num_steps = 100000

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
        log_episode = (step % 10000 == 0)
        board_save_path = None
        if log_episode:
            board_save_path = os.path.join(args.logdir, f"episode_boards_step_{step}.pkl")
        eval_stats = evaluate_policy(agent, num_episodes=100, log_episode=log_episode, save_path=board_save_path, save_board_every_step=True)
        writer.add_scalar('Eval/win_pct', eval_stats["win"], step)
        writer.add_scalar('Eval/loss_pct', eval_stats["loss"], step)
        writer.add_scalar('Eval/draw_pct', eval_stats["draw"], step)
        writer.add_scalar('Eval/invalid_pct', eval_stats["invalid"], step)
        writer.add_scalar('Eval/avg_reward', eval_stats["avg_reward"], step)
        writer.add_scalar('Eval/avg_bad_piece', eval_stats["avg_bad_piece"], step)
        writer.add_scalar('Eval/avg_threats_created', eval_stats["avg_threats_created"], step)
        writer.add_scalar('Eval/avg_threats_blocked', eval_stats["avg_threats_blocked"], step)
        writer.add_scalar('Eval/avg_turns', eval_stats["avg_turns"], step)
        print(f"Step {step}, Win: {eval_stats['win']:.2f}%, Loss: {eval_stats['loss']:.2f}%, Draw: {eval_stats['draw']:.2f}%, Invalid: {eval_stats['invalid']:.2f}%")

writer.close()



