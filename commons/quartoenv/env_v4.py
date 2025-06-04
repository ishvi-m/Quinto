import gymnasium as gym
import logging
import numpy as np
from typing import Tuple
from .base_env import QuartoBase
from .game import QuartoPiece, QUARTO_DICT
import random
from itertools import product
from ..policies import mask_function
import torch
from .encoder import MoveEncoder
from sb3_contrib import MaskablePPO
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
import wandb
from gymnasium.spaces import MultiDiscrete

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
        # observation space ~ state space
        self.observation_space = MultiDiscrete([16+1] * (16+1))  # 16 board positions + 1 hand piece
        # action space
        self.action_space = MultiDiscrete([16, 16])  # [position, piece]
        self.move_encoder = MoveEncoder()
        # self.inverse_symmetries = None  # Will be set by the training script
        self._opponent = None  # Will be set by the training script
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def opponent(self):
        """Getter method to retrieve opponent"""
        return self._opponent
    
    def update_opponent(self, new_opponent:OnPolicyAlgorithm):
        """Setter method for opponent. Implemented to perform self-play with increasingly better agents."""
        if isinstance(new_opponent, OnPolicyAlgorithm):
            del self._opponent
            self._opponent = new_opponent
        else:
            raise ValueError(f"New opponent: {new_opponent} is not an OnPolicyAlgorithm instance!")

    @property
    def _observation(self):
        """Observation is returned in the observed space composed of integers"""
        parent_obs = super()._observation
        board, current_piece = parent_obs
        board_pieces = np.fromiter(map(lambda el: 16 if el == -1 else el, board.flatten()), dtype=int)
        hand_piece = current_piece.index if current_piece else 16

        return np.append(arr=board_pieces, values=hand_piece)
    
    def reward_function(self, info: dict) -> float:
        """Computes the reward at timestep `t` given the corresponding info dictionary"""
        reward = 0.0

        # Terminal state rewards
        if info["win"]:
            reward += 10  # Win reward
            reward += 5 / info["turn"]  # Faster win bonus
        elif info["draw"]:
            reward = 2  # Draw reward
        elif info.get("loss", None):
            reward = -10  # Loss penalty
            reward -= 1.0 * info["turn"]  # Prolonged loss penalty

        # Intermediate rewards
        if info.get("threat_created", False):
            reward += 6.0  # Threat creation reward
        if info.get("threat_blocked", False):
            reward += 7.0  # Threat blocking reward
        if info.get("bad_piece", False):
            reward -= 10.0  # Bad piece penalty

        return reward
    
    def available_pieces(self)->list:
        """This function returns the pieces currently available. Those are defined as all the pieces
        available but the one each player has in hand and the ones on the board.
        
        Returns: 
            list: List of integers representing (through QUARTO_DICT) QuartoPiece(s)."""

        # retrieve the available pieces as difference between all pieces and pieces on the board
        all_pieces = set(range(16))
        current_board, current_piece = self._observation[:-1], self._observation[-1]
        # available pieces are all pieces but the ones on the board and in hand
        nonavailable_pieces = set(current_board) | {current_piece}
        available_pieces = all_pieces - nonavailable_pieces
        
        return available_pieces

    def legal_actions(self): 
        """This function returns all the legal actions given the present state encoded as int-int tuples.
        
        Yields: 
            (int, int): Tuple encoding position and piece in their integer version.
        """
        # freecells are cells with no piece inside
        freecells = np.argwhere(self.game.board == -1)
        # available pieces are those that have not been put on the board
        available_pieces = list(
            map(lambda el: QUARTO_DICT[el], self.available_pieces())) \
                if len(self.available_pieces()) > 0 \
                else [None]
        
        # a legal action is of the kind ((x,y), QuartoPiece)
        for legal_action in product(freecells, available_pieces): 
            yield self.move_encoder.encode(legal_action)

    def get_observation(self): 
        return self._observation
    
    def reset(self, *, seed=None, options=None): 
        """Resets env and returns initial observation and info dict
        
        Args:
            seed: Optional seed for random number generator
            options: Optional dictionary with additional reset options
            
        Returns:
            observation: Initial observation
            info: Empty info dict
        """
        super().reset_state()
        return self._observation, {}  # Return observation and empty info dict for Gymnasium compatibility

    def step(self, action: Tuple[int, int]):
        """Steps the environment given an action"""
        # decoding and unpacking action
        position, next = self.move_encoder.decode(action=action)

        _, _, _, truncated, info = super().step((position, next))

        # Check for threat blocking
        if self.game.threatCreated(position):
            info["threat_created"] = True

        # Check for threat blocking
        if self.game.threatBlocked(position):
            info["threat_blocked"] = True

        # Check for bad piece (piece that allows opponent to win)
        if self.game.badPieceGiven(next):
            info["bad_piece"] = True

        if not self.done:
            if self._opponent is not None:
                opponent_action, _ = self._opponent.predict(
                    observation=self._observation, 
                    action_masks = mask_function(self)
                )
                opponent_pos, opponent_piece = self.move_encoder.decode(action=opponent_action)
            else:
                # Use a random legal action if no opponent is set
                opponent_pos, opponent_piece = self.move_encoder.decode(
                    random.choice(list(self.legal_actions()))
                )
            super().step((opponent_pos, opponent_piece))
            
            if self.done: 
                info["loss"] = True
        
        reward = self.reward_function(info=info)
        return self._observation, reward, self.done, truncated, info
        

        