from .policies import *
from typing import Union
from stable_baselines3.common.callbacks import BaseCallback
import os
from typing import Tuple
import wandb
import numpy as np

class WinPercentageCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, env:ActionMasker, verbose=0, n_episodes:int=1000, logfile:str="logtraining.txt"):
        super(WinPercentageCallback, self).__init__(verbose)
        self._env = env
        self.n_episodes = n_episodes
        self.logfile = logfile
        with open(self.logfile, "w") as training_file: 
            training_file.write("# timesteps,(%) wins,(%) losses,(%) draws,(%) invalid,avg_reward,avg_threats_created,avg_threats_blocked,avg_bad_pieces,avg_turns\n")
        
        # Create trajectory log file
        self.trajectory_logfile = "trajectories.log"
        with open(self.trajectory_logfile, "w") as f:
            f.write("# Training step, Game number, Move number, Player, Action, Reward, Board state\n")
        
        self._env.reset()
        self.trajectory = []  # Store moves for trajectory
        self.last_print_step = 0  # Track last step we printed a trajectory

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `_env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        wincounter, losscounter, drawcounter, invalidcounter, matchduration = 0, 0, 0, 0, 0
        total_reward = 0  # Track total reward across episodes
        threats_created = 0  # Track total threats created
        threats_blocked = 0  # Track total threats blocked
        bad_pieces = 0  # Track total bad pieces given
        
        for episode in range(self.n_episodes):
            obs, _ = self._env.reset()  # Unpack observation and info
            done = False
            truncated = False
            self.trajectory = []  # Reset trajectory for new episode
            episode_reward = 0  # Track reward for this episode
            move_count = 0  # Track number of moves
            
            while not (done or truncated):
                # either performing a masked action or not
                if isinstance(self.model, MaskablePPO):
                    action, _ = self.model.predict(obs, action_masks = mask_function(self._env))
                else:
                    action, _ = self.model.predict(obs)
                
                # stepping the environment with the considered action 
                obs, reward, done, truncated, info = self._env.step(action=action)
                episode_reward += reward  # Accumulate reward for this episode
                move_count += 1
                
                # Store move with player information, action, board state, and info AFTER the move
                board_state = self._env.env.game.board.copy()
                self.trajectory.append((info["player"], action, board_state, info))
                
                # Log move to trajectories.log (keep this as it is for detailed logging)
                with open(self.trajectory_logfile, "a") as f:
                    f.write(f"{self.num_timesteps}, {episode}, {move_count}, {info['player']}, {action}, {reward}, {board_state.tolist()}\n")
                
                # Track strategic moves
                if info.get("threat_created", False):
                    threats_created += 1
                if info.get("threat_blocked", False):
                    threats_blocked += 1
                if info.get("bad_piece", False):
                    bad_pieces += 1
            
            total_reward += episode_reward  # Add episode reward to total
            
            # Get the observation from the environment
            parent_obs = self._env.env._observation
            
            # unpacking parent observation
            board = 16*parent_obs[:-1].reshape((4,4))
            board_image = wandb.Image(board, caption="Board in Terminal State")

            matchduration += info["turn"]/self.n_episodes

            if info["win"]: 
                wincounter += 1
            elif info.get("loss", None):
                losscounter += 1
            elif info["draw"]:
                drawcounter += 1

            # Log trajectory to WandB every 10000 steps
            if self.num_timesteps - self.last_print_step >= 10000:
                print(f"\nLogging trajectory for last episode at step {self.num_timesteps}:")
                
                # --- Console Print of Trajectory Summary ---
                print("\nGame Trajectory Summary:")
                
                last_info = {} # To store info from the last step of the trajectory for printing outcome

                for i, (player, action, board_state, info_step) in enumerate(self.trajectory):
                    # Decode the action into position and piece
                    position, next_piece_for_opponent = self._env.env.move_encoder.decode(action)
                    # Position is already in (x,y) format from decoder
                    pos_x, pos_y = position
                    
                    # Determine the piece that was actually placed in this move
                    if i == 0:
                         # For the first move, find the piece in the board state
                         non_empty = np.where(board_state != -1)
                         if non_empty[0].size > 0:
                             # Get the position from the board state where the first piece was placed
                             placed_piece = int(board_state[non_empty[0][0], non_empty[1][0]])
                             # Use the placed piece's actual position from the board state for the first move print
                             pos_x_placed, pos_y_placed = non_empty[0][0], non_empty[1][0]
                             print(f"Move {i+1}: Player: {player.upper()}, Placed: {placed_piece} at ({pos_x_placed}, {pos_y_placed}), Selected for Opponent: {next_piece_for_opponent.index}")
                         else:
                              # Should not happen in a valid first move, but handle defensively
                             print(f"Move {i+1}: Player: {player.upper()}, Placed: Unknown at Unknown Position, Selected for Opponent: {next_piece_for_opponent.index}")

                    else:
                         # For subsequent moves (i > 0), the piece placed is the one selected by the opponent in the previous move.
                         # The position is from the current action.
                         prev_action = self.trajectory[i-1][1] # Action from the previous move
                         _, piece_given_by_opponent_obj = self._env.env.move_encoder.decode(prev_action)
                         piece_given_by_opponent = piece_given_by_opponent_obj.index
                         # pos_x, pos_y from the current action are already determined at the start of the loop
                         print(f"Move {i+1}: Player: {player.upper()}, Placed: {piece_given_by_opponent} at ({pos_x}, {pos_y}), Selected for Opponent: {next_piece_for_opponent.index}")

                    # Store info from the last step
                    if i == len(self.trajectory) - 1:
                        last_info = info_step

                # Print the final info dictionary
                print("Final Game Info:", last_info)
                print("---------------------") # Separator
                # --- End Console Print ---

                # --- Logging to Trajectory Board State File ---
                trajectory_board_file = f"trajectory_board_states_step_{self.num_timesteps}.log"
                with open(trajectory_board_file, "w") as f_board:
                    f_board.write(f"# Trajectory details for step {self.num_timesteps}\n")
                    f_board.write("# Format: Move <move_number>: Placed Piece at Position\n")
                    f_board.write("---\n") # Separator

                    for i, (player, action, board_state, info_step) in enumerate(self.trajectory):
                        # Decode the action into position and piece
                        position, next_piece_for_opponent = self._env.env.move_encoder.decode(action)
                        # Position is already in (x,y) format from decoder
                        pos_x, pos_y = position

                        # Determine the piece that was actually placed in this move
                        if i == 0:
                            # For the first move, find the piece in the board state
                            non_empty = np.where(board_state != -1)
                            if non_empty[0].size > 0:
                                # Get the position from the board state where the first piece was placed
                                placed_piece = int(board_state[non_empty[0][0], non_empty[1][0]])
                                # Use the placed piece's actual position from the board state for the first move print
                                pos_x_placed, pos_y_placed = non_empty[0][0], non_empty[1][0]
                                # Print move details to the trajectory board file (simplified)
                                f_board.write(f"Move {i+1}: Placed: {placed_piece} at ({pos_x_placed}, {pos_y_placed})\n")

                                # Console print for first move
                                print(f"Move {i+1}: Player: {player.upper()}, Placed: {placed_piece} at ({pos_x_placed}, {pos_y_placed}), Selected for Opponent: {next_piece_for_opponent.index}")
                            else:
                                # Should not happen in a valid first move, but handle defensively
                                f_board.write(f"Move {i+1}: Placed: Unknown at Unknown Position\n")
                                # Console print for defensive case
                                print(f"Move {i+1}: Player: {player.upper()}, Placed: Unknown at Unknown Position, Selected for Opponent: {next_piece_for_opponent.index}")

                        else:
                            # For subsequent moves (i > 0), the piece placed is the one selected by the opponent in the previous move.
                            # The position is from the current action.
                            prev_action = self.trajectory[i-1][1] # Action from the previous move
                            _, piece_given_by_opponent_obj = self._env.env.move_encoder.decode(prev_action)
                            piece_given_by_opponent = piece_given_by_opponent_obj.index
                            # pos_x, pos_y from the current action are already determined at the start of the loop

                            # Print move details to the trajectory board file (simplified)
                            f_board.write(f"Move {i+1}: Placed: {piece_given_by_opponent} at ({pos_x}, {pos_y})\n")

                            # Console print for subsequent moves
                            print(f"Move {i+1}: Player: {player.upper()}, Placed: {piece_given_by_opponent} at ({pos_x}, {pos_y}), Selected for Opponent: {next_piece_for_opponent.index}")

                        # Write the board state to the trajectory board file
                        f_board.write(f"Board State:\n")
                        for row in board_state.tolist():
                            f_board.write(f" {row}\n")
                        f_board.write("---\n") # Separator between moves

                        # Store info from the last step
                        if i == len(self.trajectory) - 1:
                            last_info = info_step

                    # Append the full final info dictionary to the trajectory board file (after the loop)
                    f_board.write("\nFinal Game Info (raw):\n")
                    f_board.write(str(last_info) + "\n")
                    f_board.write("=====================\n") # Final Separator

                # Print the final info dictionary to the console (keeping this for quick check)
                print("Final Game Info:", last_info)
                print("---------------------") # Separator
                # --- End Console Print ---

                # Determine the game outcome from the last step's info (still used for WandB log)
                game_outcome = "Draw"
                if last_info.get("win", False):
                    game_outcome = "Win"
                elif last_info.get("loss", False):
                    game_outcome = "Loss"
                elif last_info.get("draw", False):
                     game_outcome = "Draw"
                elif last_info.get("invalid", False):
                     game_outcome = "Invalid Move"

                # Create a WandB table for the trajectory
                trajectory_table = wandb.Table(columns=["Move", "Player", "Action", "Piece Placed", "Selected for Opponent", "Position", "Threat Created", "Threat Blocked", "Bad Piece (Given)", "Board State (List)"])

                # Create a list to hold board images for the trajectory
                trajectory_images = []

                # Log the initial empty board state
                empty_board_img = wandb.Image(np.full((4, 4), -1), caption="Initial Board State")
                trajectory_images.append(empty_board_img)

                # Re-iterate through trajectory to collect data for WandB table and images
                for i, (player, action, board_state, info_step) in enumerate(self.trajectory):
                    # Decode the action into position and piece
                    position, next_piece_for_opponent = self._env.env.move_encoder.decode(action)
                    # Position is already in (x,y) format from decoder
                    pos_x, pos_y = position

                    # Determine the piece that was actually placed in this move (same logic as before)
                    if i == 0:
                         non_empty = np.where(board_state != -1)
                         if non_empty[0].size > 0:
                             pos_x_placed, pos_y_placed = non_empty[0][0], non_empty[1][0]
                             placed_piece = int(board_state[pos_x_placed, pos_y_placed])
                             # Use the position from the board state for the first move in the table and caption
                             table_pos = (pos_x_placed, pos_y_placed)
                         else:
                             placed_piece = -1
                             table_pos = (-1, -1)
                    else:
                         prev_action = self.trajectory[i-1][1]
                         _, piece_given_by_opponent_obj = self._env.env.move_encoder.decode(prev_action)
                         placed_piece = piece_given_by_opponent_obj.index
                         table_pos = position # Use position from action decode for subsequent moves

                    # Add data to the table, including strategic move flags from info_step
                    trajectory_table.add_data(
                        i + 1,
                        player,
                        str(action), # Log raw action for reference
                        placed_piece,
                        next_piece_for_opponent.index,
                        str(table_pos), # Include position here
                        info_step.get("threat_created", False), # Include threat created flag
                        info_step.get("threat_blocked", False), # Include threat blocked flag
                        info_step.get("bad_piece", False), # Include bad piece flag
                        board_state.tolist() # Log board state as a list of lists
                    )

                    # Create an image for the board state after this move
                    if i == 0:
                        # Caption for the first move (Agent's first play)
                        board_img = wandb.Image(board_state, caption=f"Move {i+1} - Player: {player.upper()}, Placed: {placed_piece} at ({table_pos[0]}, {table_pos[1]}), Selected for Opponent: {next_piece_for_opponent.index} - Board State")
                    else:
                        # Caption for subsequent moves (i > 0)
                        # The piece placed in this move is the one selected by the opponent in the previous move.
                        # The position is from the current action.
                        board_img = wandb.Image(board_state, caption=f"Move {i+1} - Player: {player.upper()}, Placed: {placed_piece} at ({table_pos[0]}, {table_pos[1]}), Selected for Opponent: {next_piece_for_opponent.index} - Board State")

                    trajectory_images.append(board_img)

                # Log the table and images to WandB, including the game outcome
                wandb.log({
                    "Game Trajectory Table": trajectory_table,
                    "Game Trajectory Boards": trajectory_images,
                    "Game Outcome": game_outcome # Log the final outcome
                }, step=self.num_timesteps)

                self.last_print_step = self.num_timesteps

        # Calculate averages
        avg_reward = total_reward / self.n_episodes
        avg_threats_created = threats_created / self.n_episodes
        avg_threats_blocked = threats_blocked / self.n_episodes
        avg_bad_pieces = bad_pieces / self.n_episodes

        # Calculate rates
        win_rate = 100 * wincounter / self.n_episodes
        loss_rate = 100 * losscounter / self.n_episodes
        draw_rate = 100 * drawcounter / self.n_episodes

        # resettin the environment at the end of testing phase
        self._env.reset()
            
        with open(self.logfile, "a") as training_file: 
            training_file.write(
                "\n{}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(
                    self.num_timesteps, 
                    win_rate,
                    loss_rate,
                    draw_rate,
                    100 * invalidcounter / self.n_episodes,
                    avg_reward,
                    avg_threats_created,
                    avg_threats_blocked,
                    avg_bad_pieces,
                    matchduration
                )
            )

        # Log to tensorboard
        self.logger.record("win_rate", win_rate)
        self.logger.record("loss_rate", loss_rate)
        self.logger.record("draw_rate", draw_rate)
        self.logger.record("avg_reward", avg_reward)
        self.logger.record("avg_threats_created", avg_threats_created)
        self.logger.record("avg_threats_blocked", avg_threats_blocked)
        self.logger.record("avg_bad_pieces", avg_bad_pieces)
        self.logger.record("avg_turns", matchduration)
        self.logger.dump(self.num_timesteps)

        # Log to wandb
        wandb.log({
            "Win(%)": win_rate,
            "Loss(%)": loss_rate,
            "Draw(%)": draw_rate,
            "Invalid(%)": 100 * invalidcounter / self.n_episodes,
            "Game Turns": matchduration,
            "Average Reward": avg_reward,
            "Avg Threats Created": avg_threats_created,
            "Avg Threats Blocked": avg_threats_blocked,
            "Avg Bad Pieces": avg_bad_pieces,
            "Board": board_image
        })
        
        return True

class UpdateOpponentCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, checkpoints_dir:str="checkpoints/", verbose:int=0):
        super().__init__(verbose)
        self.checkpoint_dir = checkpoints_dir

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `_env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        # access last element in listdir - usually mostly trained agent ~ best available
        candidate_model = sorted(os.listdir(self.checkpoint_dir), key=lambda name: int(name.split(".")[0].split("_")[1]))[-1]
        candidate = MaskablePPO.load("checkpoints/" + candidate_model)
        # updating the current opponent with new candidate
        self.model.env.envs[0].update_opponent(candidate)

        return True

def preprocess_policy(_env:gym.Env, policy:Union[OnPolicyAlgorithm, str])->Union[MaskedRandomPolicy, RandomPolicy, OnPolicy]:
        """This function pre-processes the input policy returning an instance of the policy that
        exposes a `train_policy` method.

        Args: 
            policy (Union[OnPolicyAlgorithm, str]): Either an OnPolicyAlgorithm (from stable_baselines3) or a string
                                                    identifying one of the dummy policies here implemented for testing.
        
        Raises: 
            ValueError: When policy is a string not in ["fair-random", "pure-random"].
            Warning: When policy is an OnPolicyAlgorithm (from stable_baselines3) that is not among the trained algorithms here. 
                     Implemented algorithms are in __init__ function of policies module.

        Returns: 
            Union[MaskedRandomPolicy, RandomPolicy, QuartoOnPolicy]: Preprocessed Policy Object.
        """
        if isinstance(policy, str): 
            if policy.lower() not in ["fair-random", "pure-random"]:
                raise ValueError(f"Policy {policy} not in ['fair-random', 'pure-random']!")
            else:
                if policy.lower() == "fair-random": 
                    actual_policy = MaskedRandomPolicy(_env=_env)
                elif policy.lower() == "pure-random": 
                    actual_policy = RandomPolicy(_env=_env)
        
        elif isinstance(policy, OnPolicyAlgorithm):
            if type(policy) not in Algorithms:
                print("Policy algorithm is not among already implemented ones!")
                raise Warning(f"Implemented algorithms are: {'/'.join([AlgoDict[algo] for algo in Algorithms])}")
            actual_policy = OnPolicy(_env=_env, model=policy)
        
        return actual_policy

class QuartoPolicy:
    "Base class wrapping various policies for model evaluation"
    def __init__(self, _env:gym.Env, policy:Union[OnPolicyAlgorithm, str]="pure-random", verbose:int=1):
        self._env = _env
        self._policy = preprocess_policy(_env=self._env, policy=policy)
        self.verbose = verbose

    @property
    def _env(self): 
        return self._env
    
    def test_policy(self, n_episodes:int=1_000, verbose:int=1)->None: 
        """Tests considered policy over `n_episodes`"""
        return self._policy.test_policy(n_episodes=n_episodes, verbose=verbose)

def anti90(x:int, y:int)->tuple:
    """Maps points after a 90-degrees rotation to their original representation."""
    return y, 3-x

def clock_rotation(x:int, y:int, degree:float)->tuple: 
    """Applies clock-wise rotation on the vector whose components are x and y.
    This is done in the sake of restoring the points obtained in the original space.
    
    Args:
        x (int): x-coordinate of the point considered.
        y (int): y-coordinate of the point considered.
        degree (float): degree in which to perform the rotation. Since the aim of this function is reconstruction, 
                        rotation is `clockwise`.
    
    Returns:
        (np.array): point in the old, non-rotated, space.
    """
    if degree==90.:
        return anti90(x,y)
    elif degree==180.:
        return anti90(*clock_rotation(x,y, degree=90))
    elif degree==270.:
        return anti90(*clock_rotation(x,y, degree=180))
    else:
        raise ValueError("Rotations for angles >270 are not yet implemented!")

def antiflip(x,y, verse:str):
    """Applies anti-flipping operation on the vector whose components are x and y.
    This is done in the sake of restoring the points obtained in the original space.
    
    Args:
        x (float): x-coordinate of the point considered.
        y (float): y-coordinate of the point considered.
        verse (str): verse in which to rotate once more the point considered.
    
    Returns:
        (np.array): point in the new, un-flipped, space.
    """
    if verse.lower()=="horizontal" or verse.lower()=="h":
        return np.array([3-x,y])
    elif verse.lower()=="vertical" or verse.lower()=="v":
        return np.array([x,3-y])
    else:
        raise ValueError("Anti-Flip operations are diagonally and horizontally only!")

class QuartoSymmetries:
    def __init__(self):
        # Define a list of all possible symmetries in a game of Quarto
        self.symmetries = {
            # identity
            "identity": lambda x: x,
            # Rotate 90 degrees
            "rot90": lambda x: np.rot90(x, k=1),
            # Rotate 180 degrees
            "rot180": lambda x: np.rot90(x, k=2),
            # Rotate 270 degrees
            "rot270": lambda x: np.rot90(x, k=3)
            # Reflect horizontally
            # "h_flip": lambda x: np.fliplr(x),
            # Reflect vertically
            # "v_flip": lambda x: np.flipud(x)
        }
        # Define inverse symmetries
        self.inverse_symmetries = {
            # identity
            "identity": lambda x,y: (x,y), 
            # Rotate -90 degrees (270 degrees)
            "rot90": lambda x,y : clock_rotation(x,y, degree=90),
            # Rotate -180 degrees (180 degrees)
            "rot180": lambda x,y: clock_rotation(x,y, degree=180),
            # Rotate -270 degrees (90 degrees)
            "rot270": lambda x,y: clock_rotation(x,y, degree=270)
            # Reflect horizontally, once more
            # "h_flip": lambda x,y: antiflip(x,y, verse="horizontal"),
            # Reflect vertically, once more
            # "v_flip": lambda x,y: antiflip(x,y, verse="vertical")
        }

    def apply_symmetries(self, board:np.array)->Tuple[np.array, list, list]:
        """This function applies the Quarto game symmetries to a given board, returning the board considered
        and the function that can be used to re-obtain original form (i.e., the function that when applied on apply_symmetries output
        would output board input once more)

        Args: 
            board (np.array): original board to translate to its canonical form
        
        Returns: 
            Tuple[np.array, list, list]: board in the canonical form, list of functions used for canonization and list of functions that 
            can be used to revert canonization
        """
        inv_symmetries = []
        applied_symmetries = []
        # applies symmetries
        canonical_board = board
        for name, symmetry in self.symmetries.items():
            # applies the symmetry
            sym_board = symmetry(board)
            # induces an order among alternative representations
            if sym_board[0,0] < canonical_board[0,0]:
                canonical_board = sym_board
                # stores the new symmetrical representation
                applied_symmetries.append(name)
                inv_symmetries.append(self.inverse_symmetries[name])

        if not inv_symmetries:
            inv_symmetries.append(self.inverse_symmetries["identity"])
        return canonical_board, [inv_symmetries[-1]]
