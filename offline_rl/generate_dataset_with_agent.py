import argparse
import os
import pickle
import sys
import numpy as np

# Ensure project root is on PYTHONPATH so relative imports work even when running from elsewhere
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from sb3_contrib.ppo_mask import MaskablePPO
from commons.policies.onpolicy_wrapper import mask_function
from sb3_contrib.common.wrappers import ActionMasker

# Import environments lazily to avoid circular deps
from commons.quartoenv.env import RandomOpponentEnv  # v0
from commons.quartoenv.env_v4 import CustomOpponentEnv_V4  # v4


def make_env(version: str):
    """Return an environment instance according to the requested version."""
    if version == "v0":
        env = RandomOpponentEnv()
    elif version == "v4":
        env = CustomOpponentEnv_V4()
    else:
        raise ValueError(f"Unknown env_version '{version}'. Supported: ['v0', 'v4']")
    return env


def main():
    parser = argparse.ArgumentParser(description="Generate an offline dataset using a pretrained MaskedPPO agent.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the .zip file of the pretrained MaskedPPO agent")
    parser.add_argument("--env_version", type=str, choices=["v0", "v4"], default="v4", help="Environment version to roll out in")
    parser.add_argument("--num_episodes", type=int, default=10000, help="Number of episodes to collect")
    parser.add_argument("--output_path", type=str, default="offline_quinto_dataset.pkl", help="Where to store the resulting dataset")
    parser.add_argument("--use_action_masking", action="store_true", help="Wrap the environment with action masking (required for MaskedPPO)")
    args = parser.parse_args()

    env = make_env(args.env_version)

    # If we wrap with ActionMasker, we still need easy access to the underlying
    # environment (for helper methods like `legal_actions`).
    if args.use_action_masking:
        env = ActionMasker(env, mask_function)

    def _base_env(e):
        """Return the innermost env (unwrap helpers / mask wrappers)."""
        return e.env if hasattr(e, "env") else e

    # Load the pretrained MaskedPPO agent
    model = MaskablePPO.load(args.model_path, env=env, custom_objects={"learning_rate": 0.0, "clip_range": 0.0})

    dataset = []

    for ep in range(args.num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            # Retrieve action mask if required
            if args.use_action_masking:
                action_masks = mask_function(env)
                action, _ = model.predict(obs, action_masks=action_masks)
            else:
                action, _ = model.predict(obs)

            base_env = _base_env(env)
            legal_actions = [pos * 16 + piece for (pos, piece) in base_env.legal_actions() if piece is not None]
            next_obs, reward, done, truncated, info = env.step(action)
            next_legal_actions = [pos * 16 + piece for (pos, piece) in _base_env(env).legal_actions() if piece is not None]

            dataset.append(
                {
                    "state": obs,
                    "action": action,
                    "reward": reward,
                    "next_state": next_obs,
                    "done": done,
                    "info": info,
                    "legal_actions": legal_actions,
                    "next_legal_actions": next_legal_actions,
                }
            )
            obs = next_obs

        if (ep + 1) % 100 == 0:
            print(f"Collected {ep + 1}/{args.num_episodes} episodes", flush=True)

    # Save dataset
    with open(args.output_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"Saved dataset with {len(dataset)} transitions to {args.output_path}")


if __name__ == "__main__":
    main() 