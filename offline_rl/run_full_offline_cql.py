import subprocess
import sys
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter

# 1. Generate the dataset
print("Generating dataset with generate_dataset.py...")
subprocess.run([sys.executable, "offline_rl/generate_dataset.py"], check=True)

# 2. Run offline CQL training (on GPU if available)
print("Running offline CQL training with offline_cql.py...")

# Check for GPU
if torch.cuda.is_available():
    print("CUDA is available. Training will use GPU.")
else:
    print("CUDA is NOT available. Training will use CPU.")

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, default='runs/offline_cql_cs224r', help='TensorBoard log directory')
args = parser.parse_args()

subprocess.run([sys.executable, "offline_rl/offline_cql.py", "--logdir", args.logdir], check=True)

print("\nAll done! To monitor training, run:")
print("  tensorboard --logdir runs/offline_cql_cs224r\n")
print("If you want to check GPU usage during training, run:")
print("  watch -n 1 nvidia-smi\n")

writer = SummaryWriter(log_dir=args.logdir) 