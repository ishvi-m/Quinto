import gym
import logging
import numpy as np
from typing import Tuple
from .game import QuartoGame, QuartoPiece, QUARTO_DICT
from itertools import product

logger = logging.getLogger(__name__)

class QuartoBase(gym.Env):
    def __init__(self): 
        
        self.game = QuartoGame()
        self.turns = 0
        self.piece = None
        self.broken = False
        self.EMPTY = 0
        self.metadata = {'render.modes':['human', 'terminal']}
        
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
        """ State of the game after the move.
        """
        return (self.game.board, self.piece)

    @property
    def done(self):
        return self.broken or self.game.game_over or self.game.draw

    def reset_state(self):
        """This function takes care of resetting the environment to initial state, 
        i.e. empty board with no pieces on."""
        self.game = QuartoGame()
        self.turns = 0
        self.piece = None
        self.broken = False
        
        # Update lifetime stats with episode stats before resetting
        self.lifetime_threats_created += self.total_threats_created
        self.lifetime_threats_blocked += self.total_threats_blocked
        self.lifetime_center_positions += self.total_center_positions
        self.lifetime_bad_pieces += self.total_bad_pieces
        self.lifetime_losses += self.total_losses
        self.total_episodes += 1

        # Log lifetime statistics at episode end
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

    def step(self, action:Tuple[tuple, QuartoPiece])->Tuple:
        """This function steps the environment considering the given action"""
        # sparse rewards - no reward but in terminal states
        # TODO: Might be interesting to reward negatively the agent at each turn to encourage
        #       winning early. Idea
        # CS224R extension on Quinto: dense rewards!
        
        reward = 0
        # increment number of turns
        self.turns += 1
        info = {'turn': self.turns,
                'invalid': False,
                'win': False,
                'draw': False,
                'threat_created': False}
        
        if self.done:
            logger.warn("Actually already done")
            return self._observation, reward, self.done, info

        position, next = action
        logger.debug(f"Received: position: {position}, next: {next}")

        # self.piece stores the piece that has to be positioned on the board. 
        # self.piece is None at first turn, i.e. at the beginning of the game.
        
        if self.piece is not None:
            # play the current piece on the board
            valid = self.game.play(piece=self.piece, position=position, next_piece=next)

            if not valid:
                # Invalid move
                reward = -200
                self.broken = True  # boolean indicator indicating when invalid action is performed
                info['invalid'] = True

            elif self.game.threatCreated(position):
                info['threat_created'] = True
                self.total_threats_created += 1
                logger.info(f"Threat created at position {position} (Total: {self.total_threats_created})")
            
            # check if played move makes the agent win
            elif self.game.game_over:
                # We just won!
                info['win'] = True
            
            # check draw
            elif self.game.draw:
                info['draw'] = True
            else:
                # a valid move was played
                pass

            # Track additional statistics in info
            if position in [(1,1), (1,2), (2,1), (2,2)]:
                info['center_position'] = True
                self.total_center_positions += 1
                logger.info(f"Center position used at {position} (Total: {self.total_center_positions})")

            if self.game.threatBlocked(position):
                info['threat_blocked'] = True
                self.total_threats_blocked += 1
                logger.info(f"Threat blocked at position {position} (Total: {self.total_threats_blocked})")

            if self.game.badPieceGiven(next):
                info['bad_piece'] = True
                self.total_bad_pieces += 1
                logger.info(f"Bad piece given: {next} (Total: {self.total_bad_pieces})")

            # Add aggregated stats to info
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
        
        # Process the next piece
        self.piece = next
        
        return self._observation, reward, self.done, info

    def render(self, mode:str="human", **kwargs):
        "Renders board printing to standard output pieces in their encoding"
        for row in self.game.board:
            s = ""
            for piece in row:
                if piece is None:
                    s += ". "
                else:
                    s += str(piece) + " "
            print(s)

        print(f"Next: {str(self.piece.index)}, Free: {'/'.join(str(p.index) for p in self.available_pieces())}")

    def __del__(self):
        self.close()
