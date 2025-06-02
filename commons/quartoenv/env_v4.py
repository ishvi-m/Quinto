import gym
import logging
import numpy as np
from typing import Tuple
from .base_env import QuartoBase
from .game import QuartoPiece, QUARTO_DICT
import random
from itertools import product
from ..policies import mask_function
import torch

logger = logging.getLogger(__name__)

class CustomOpponentEnv_V4(QuartoBase):
    """
    Environment version 4 that implements comprehensive reward functions:
    - Threat Creation: +1 for creating a line of 3 with shared attribute
    - Threat Blocking: +0.5 for blocking opponent's potential win
    - Bad Piece Penalty: -0.5 for giving opponent a winning piece
    - Center Preference: +0.1 for central positions
    - Faster win bonus: +10/num_turns
    - Prolonged loss penalty: -1 * num_turns if losing
    - Win reward: +10
    - Loss penalty: -10
    - Draw: 0
    """

    def __init__(self):
        super().__init__()
        self.move_encoder = None  # Will be set by the training script
        self.inverse_symmetries = None  # Will be set by the training script
        self._opponent = None  # Will be set by the training script
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Pre-compute center positions for faster checking
        self.center_positions = set([(1, 1), (1, 2), (2, 1), (2, 2)])

    def update_opponent(self, new_opponent):
        """Update the opponent model"""
        self._opponent = new_opponent

    def reward_function(self, info: dict) -> float:
        """Computes the reward at timestep `t` given the corresponding info dictionary"""
        reward = 0.0

        # Terminal state rewards
        if info["win"]:
            reward += 10  # Win reward
            reward += 5 / info["turn"]  # Faster win bonus
        elif info["draw"]:
            reward = 1  # Draw reward
        elif info.get("loss", None):
            reward = -10  # Loss penalty
            reward -= 0.5 * info["turn"]  # Prolonged loss penalty

        # Intermediate rewards
        if info.get("threat_created", False):
            reward += 1.0  # Threat creation reward
        if info.get("threat_blocked", False):
            reward += 0.5  # Threat blocking reward
        if info.get("bad_piece", False):
            reward -= 0.5  # Bad piece penalty
        if info.get("center_position", False):
            reward += 0.1  # Center preference reward

        return reward

    def step(self, action: Tuple[int, int]):
        """Steps the environment given an action"""
        # Decoding and unpacking action
        position, next = self.move_encoder.decode(action=action)
        
        # Apply symmetries if needed
        if self.inverse_symmetries:
            for inverse_symmetry in self.inverse_symmetries:
                position = inverse_symmetry(*position)

        # Agent's move
        _, _, _, info = super().step((position, next))

        # Check for center position (using pre-computed set for O(1) lookup)
        if position in self.center_positions:
            info["center_position"] = True

        # Check for threat blocking
        if self.game.threatBlocked(position):
            info["threat_blocked"] = True

        # Check for bad piece (piece that allows opponent to win)
        if self.game.badPieceGiven(next):
            info["bad_piece"] = True

        if not self.done:
            # Opponent's reply
            opponent_action, _ = self._opponent.predict(
                observation=self._observation,
                action_masks=mask_function(self)
            )
            opponent_pos, opponent_piece = self.move_encoder.decode(action=opponent_action)
            
            # Apply symmetries to opponent's move if needed
            if self.inverse_symmetries:
                for inverse_symmetry in self.inverse_symmetries:
                    opponent_pos = inverse_symmetry(*opponent_pos)
            
            # Step environment with opponent's move
            _, reward, _, info = super().step((opponent_pos, opponent_piece))
            
            if self.done:
                info["loss"] = True

        # Calculate reward using our reward function
        reward = self.reward_function(info)
        
        return self._observation, reward, self.done, info 