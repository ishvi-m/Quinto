import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
from commons.quartoenv.env_v4 import CustomOpponentEnv_V4

env = CustomOpponentEnv_V4()
num_episodes = 10000
dataset = []

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    while not done:
        # Choose action (random, rule-based, etc.)
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        
        # Store transition
        dataset.append({
            "state": obs,
            "action": action,
            "reward": reward,
            "next_state": next_obs,
            "done": done,
            "info": info
        })
        obs = next_obs

# Save dataset
with open("offline_quinto_dataset.pkl", "wb") as f:
    pickle.dump(dataset, f)

print(dataset[0])
print(dataset[100])
print(dataset[-1])

threats = [d for d in dataset if d['info'].get('threat', False)]
print(f"Number of threat transitions: {len(threats)}")
wins = [d for d in dataset if d['info'].get('win', False)]
print(f"Number of win transitions: {len(wins)}")
