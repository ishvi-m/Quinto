from .base_env import QuartoBase
from .encoder import MoveEncoder
from gym.spaces import MultiDiscrete
import logging
import numpy as np
from typing import Tuple
from .game import QUARTO_DICT
from itertools import product
import random

logger = logging.getLogger(__name__)

class RandomOpponentEnv(QuartoBase):
    """
    Quarto Env supporting state and action encoding
    That's a subclass and not a wrapper.
    Moreover, this class also models a random opponent always playing valid moves.
    """
    def __init__(self):
        """
        State space: describes board + player hand 
                           board described as 16 cells in which one can find a piece (0-15) or nothing (16)
                           same goes with hand, either a piece (0-15) or nothing (16)
        
        Actions space: described as move played (cell played) and piece chosen from still
                       availables. 
                       16: moves that can be chosen; 16: pieces that can be played
        """
        super().__init__()
        # observation space ~ state space
        self.observation_space = MultiDiscrete([16+1] * (16+1))
        # action space
        self.action_space = MultiDiscrete([16, 16])
        # move encoder to take care of turning tuples/objects into integers
        self.move_encoder = MoveEncoder()

        # Episode statistics counters
        self.total_threats_created = 0
        self.total_threats_blocked = 0
        self.total_center_positions = 0
        self.total_bad_pieces = 0
        self.total_losses = 0

        # Lifetime statistics (across all episodes)
        self.lifetime_threats_created = 0
        self.lifetime_threats_blocked = 0
        self.lifetime_center_positions = 0
        self.lifetime_bad_pieces = 0
        self.lifetime_losses = 0
        self.total_episodes = 0

    @property
    def _observation(self):
        """Observation is returned in the observed space composed of integers"""
        # accessing parent observation (board, current_piece)
        parent_obs = super()._observation
        # unpacking parent observation
        board, current_piece = parent_obs
        # turning parent observation into a point of the observation space here defined
        board_pieces = np.fromiter(map(lambda el: 16 if el == -1 else el, board.flatten()), dtype=int)
        hand_piece = current_piece.index if current_piece else 16
        
        return np.append(arr=board_pieces, values=hand_piece)
    
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
        freecells = self.game.free_spots
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
    
    def reset(self): 
        """Resets env"""
        # Update lifetime stats with episode stats before resetting
        self.lifetime_threats_created += self.total_threats_created
        self.lifetime_threats_blocked += self.total_threats_blocked
        self.lifetime_center_positions += self.total_center_positions
        self.lifetime_bad_pieces += self.total_bad_pieces
        self.lifetime_losses += self.total_losses
        self.total_episodes += 1

        # Log lifetime statistics at episode end
        if self.total_episodes > 0:  # Only log if we've completed at least one episode
            logger.info(f"\nLifetime Statistics (After {self.total_episodes} episodes):")
            logger.info(f"Average threats created per episode: {self.lifetime_threats_created/self.total_episodes:.2f}")
            logger.info(f"Average threats blocked per episode: {self.lifetime_threats_blocked/self.total_episodes:.2f}")
            logger.info(f"Average center positions per episode: {self.lifetime_center_positions/self.total_episodes:.2f}")
            logger.info(f"Average bad pieces per episode: {self.lifetime_bad_pieces/self.total_episodes:.2f}")
            logger.info(f"Win rate: {(self.total_episodes - self.lifetime_losses)/self.total_episodes:.2%}\n")

        # Reset episode statistics counters
        self.total_threats_created = 0
        self.total_threats_blocked = 0
        self.total_center_positions = 0
        self.total_bad_pieces = 0
        self.total_losses = 0

        super().reset_state()
        return self._observation
    
    def step(self, action): 
        """This function steps the environment considering the given action"""
        # Decoding and unpacking action
        position, next = self.move_encoder.decode(action=action)
        
        # Agent's move
        _, reward, done, info = super().step((position, next))

        # Track statistics
        if info.get('threat_created', False):
            self.total_threats_created += 1
            logger.info(f"Threat created at position {position} (Total: {self.total_threats_created})")

        # Check for center position
        if position in [(1,1), (1,2), (2,1), (2,2)]:
            self.total_center_positions += 1
            info['center_position'] = True
            logger.info(f"Center position used at {position} (Total: {self.total_center_positions})")

        # Check for threat blocking
        if self.game.threatBlocked(position):
            self.total_threats_blocked += 1
            info['threat_blocked'] = True
            logger.info(f"Threat blocked at position {position} (Total: {self.total_threats_blocked})")

        # Check for bad piece
        if self.game.badPieceGiven(next):
            self.total_bad_pieces += 1
            info['bad_piece'] = True
            logger.info(f"Bad piece given: {next} (Total: {self.total_bad_pieces})")

        if done:
            # Random opponent's move
            available_actions = list(self.legal_actions())
            if available_actions:  # if there are legal actions
                opponent_action = random.choice(available_actions)
                opponent_pos, opponent_piece = self.move_encoder.decode(action=opponent_action)
                
                # Step environment with opponent's move
                _, reward, done, info = super().step((opponent_pos, opponent_piece))
                
                if done:
                    info["loss"] = True
                    self.total_losses += 1
                    logger.info(f"Game lost (Total losses: {self.total_losses})")

            # Log episode statistics at the end of game
            logger.info("Final game statistics:")
            logger.info(f"This episode:")
            logger.info(f"  Threats created: {self.total_threats_created}")
            logger.info(f"  Threats blocked: {self.total_threats_blocked}")
            logger.info(f"  Center positions used: {self.total_center_positions}")
            logger.info(f"  Bad pieces given: {self.total_bad_pieces}")
            logger.info(f"  Game {'lost' if info.get('loss', False) else 'won'}")

            # Calculate lifetime totals (including current episode)
            curr_lifetime_threats = self.lifetime_threats_created + self.total_threats_created
            curr_lifetime_blocks = self.lifetime_threats_blocked + self.total_threats_blocked
            curr_lifetime_centers = self.lifetime_center_positions + self.total_center_positions
            curr_lifetime_bad = self.lifetime_bad_pieces + self.total_bad_pieces
            curr_lifetime_losses = self.lifetime_losses + (1 if info.get('loss', False) else 0)
            curr_episodes = self.total_episodes + 1

            logger.info(f"\nLifetime statistics (including this episode):")
            logger.info(f"  Average threats created: {curr_lifetime_threats/curr_episodes:.2f}")
            logger.info(f"  Average threats blocked: {curr_lifetime_blocks/curr_episodes:.2f}")
            logger.info(f"  Average center positions: {curr_lifetime_centers/curr_episodes:.2f}")
            logger.info(f"  Average bad pieces: {curr_lifetime_bad/curr_episodes:.2f}")
            logger.info(f"  Win rate: {(curr_episodes - curr_lifetime_losses)/curr_episodes:.2%}")

        # Add all statistics to info dictionary
        info.update({
            'total_threats_created': self.total_threats_created,
            'total_threats_blocked': self.total_threats_blocked,
            'total_center_positions': self.total_center_positions,
            'total_bad_pieces': self.total_bad_pieces,
            'total_losses': self.total_losses,
            'lifetime_threats_created': self.lifetime_threats_created + self.total_threats_created,
            'lifetime_threats_blocked': self.lifetime_threats_blocked + self.total_threats_blocked,
            'lifetime_center_positions': self.lifetime_center_positions + self.total_center_positions,
            'lifetime_bad_pieces': self.lifetime_bad_pieces + self.total_bad_pieces,
            'lifetime_losses': self.lifetime_losses + self.total_losses,
            'total_episodes': self.total_episodes
        })

        return self._observation, reward, done, info
