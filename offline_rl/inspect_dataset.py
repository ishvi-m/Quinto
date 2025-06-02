import pickle

with open("offline_quinto_dataset.pkl", "rb") as f:
    data = pickle.load(f)

print(f"Number of transitions: {len(data)}")
print("First transition keys:", data[0].keys())
print("Sample transition (first):")
for k, v in data[0].items():
    print(f"  {k}: {type(v)}")

# Count the number of episodes (number of transitions where done == True)
num_episodes = sum(1 for d in data if d['done'])
print(f"Number of episodes (rollouts): {num_episodes}")
