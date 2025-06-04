import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
from commons.quartoenv.env_v4 import CustomOpponentEnv_V4

env = CustomOpponentEnv_V4()
num_episodes = 10000
dataset = []

for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    while not done:
        # Choose action (random, rule-based, etc.)
        action = env.action_space.sample()
        legal_actions = [pos * 16 + piece for (pos, piece) in env.legal_actions()]
        next_obs, reward, done, truncated, info = env.step(action)
        next_legal_actions = [pos * 16 + piece for (pos, piece) in env.legal_actions()]
        
        # Store transition
        dataset.append({
            "state": obs,
            "action": action,
            "reward": reward,
            "next_state": next_obs,
            "done": done,
            "info": info,
            "legal_actions": legal_actions,
            "next_legal_actions": next_legal_actions
        })
        obs = next_obs

# Save dataset
with open("offline_quinto_dataset.pkl", "wb") as f:
    pickle.dump(dataset, f)

print(dataset[0])
print(dataset[100])
print(dataset[-1])

threats = [d for d in dataset if d['info'].get('threat_created', False)]
print(f"Number of threat_created transitions: {len(threats)}")
wins = [d for d in dataset if d['info'].get('win', False)]
print(f"Number of win transitions: {len(wins)}")

blocked_threats = [d for d in dataset if d['info'].get('threat_blocked', False)]
print(f"Number of threat_blocked transitions: {len(blocked_threats)}")
