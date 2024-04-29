"""
Environment.
"""
import random
import datetime
import numpy as np
from tqdm import trange
from pathlib import Path
from warnings import warn
from time import time, sleep
from datetime import datetime
from world.helpers import save_results, action_to_direction

try:
    from agents import BaseAgent
    from world.grid import Grid
    from world.gui import GUI
    from world.path_visualizer import visualize_path
except ModuleNotFoundError:
    from os import path
    from os import pardir
    import sys

    root_path = path.abspath(path.join(
        path.join(path.abspath(__file__), pardir), pardir)
    )

    if root_path not in sys.path:
        sys.path.append(root_path)

    from agents import BaseAgent
    from world.grid import Grid
    from world.gui import GUI
    from world.path_visualizer import visualize_path

class Environment:
    def __init__(self,
                 grid_fp: Path,
                 no_gui: bool = False,
                 sigma: float = 0.,
                 agent_start_pos: tuple[int, int] = None,
                 reward_fn: callable = None,
                 target_fps: int = 30,
                 random_seed: int | float | str | bytes | bytearray | None = 0):
        
        """Creates the Grid Environment for the Reinforcement Learning robot
        from the provided file.

        This environment follows the general principles of reinforcment
        learning. It can be thought of as a function E : action -> observation
        where E is the environment represented as a function.

        Args:
            grid_fp: Path to the grid file to use.
            no_gui: True if no GUI is desired.
            sigma: The stochasticity of the environment. The probability that
                the agent makes the move that it has provided as an action is
                calculated as 1-sigma.
            agent_start_pos: Tuple where each agent should start.
                If None is provided, then a random start position is used.
            reward_fn: Custom reward function to use. 
            target_fps: How fast the simulation should run if it is being shown
                in a GUI. If in no_gui mode, then the simulation will run as fast as
                possible. We may set a low FPS so we can actually see what's
                happening. Set to 0 or less to unlock FPS.
            random_seed: The random seed to use for this environment. If None
                is provided, then the seed will be set to 0.
        """
        random.seed(random_seed)

        # Initialize Grid
        if not grid_fp.exists():
            raise FileNotFoundError(f"Grid {grid_fp} does not exist.")
        else:
            self.grid_fp = grid_fp

        # Initialize other variables
        self.agent_start_pos = agent_start_pos
        self.terminal_state = False
        self.sigma = sigma
              
        # Set up reward function
        if reward_fn is None:
            warn("No reward function provided. Using default reward.")
            self.reward_fn = self._default_reward_function
        else:
            self.reward_fn = reward_fn

        # GUI specific code: Set up the environment as a blank state.
        self.no_gui = no_gui
        if target_fps <= 0:
            self.target_spf = 0.
        else:
            self.target_spf = 1. / target_fps
        self.gui = None

    def _reset_info(self) -> dict:
        """Resets the info dictionary.

        info is a dict with information of the most recent step
        consisting of whether the target was reached or the agent
        moved and the updated agent position.
        """
        return {"target_reached": False,
                "agent_moved": False,
                "actual_action": None}
    
    @staticmethod
    def _reset_world_stats() -> dict:
        """Resets the world stats dictionary.

        world_stats is a dict with information about the 
        environment since last env.reset(). Basically, it
        accumulates information.
        """
        return {"cumulative_reward": 0,
                "total_steps": 0,
                "total_agent_moves": 0,
                "total_failed_moves": 0,
                "total_targets_reached": 0,
                }

    def _initialize_agent_pos(self):
        """Initializes agent position from the given location or
        randomly chooses one if None was given.
        """

        if self.agent_start_pos is not None:
            pos = (self.agent_start_pos[0], self.agent_start_pos[1])
            if self.grid[pos] == 0:
                # Cell is empty. We can place the agent there.
                self.agent_pos = pos
            else:
                raise ValueError(
                    "Attempted to place agent on top of obstacle, delivery"
                    "location or charger")
        else:
            # No positions were given. We place agents randomly.
            warn("No initial agent positions given. Randomly placing agents "
                 "on the grid.")
            # Find all empty locations and choose one at random
            zeros = np.where(self.grid == 0)
            idx = random.randint(0, len(zeros[0]) - 1)
            self.agent_pos = (zeros[0][idx], zeros[1][idx])


    def reset(self, **kwargs) -> tuple[int, int]:
        """Reset the environment to an initial state.

        You can fit it keyword arguments which will overwrite the 
        initial arguments provided when initializing the environment.

        Args:
            **kwargs: possible keyword options are the same as those for
                the environment initializer.
        Returns:
             initial state.
        """
        for k, v in kwargs.items():
            # Go through each possible keyword argument.
            match k:
                case "grid_fp":
                    self.grid_fp = v
                case "agent_start_pos":
                    self.agent_start_pos = v
                case "no_gui":
                    self.no_gui = v
                case "target_fps":
                    self.target_spf = 1. / v
                case _:
                    raise ValueError(f"{k} is not one of the possible "
                                     f"keyword arguments.")
        
        # Reset variables
        self.grid = Grid.load_grid(self.grid_fp).cells
        self._initialize_agent_pos()
        self.terminal_state = False
        self.info = self._reset_info()
        self.world_stats = self._reset_world_stats()

        # GUI specific code
        if not self.no_gui:
            self.gui = GUI(self.grid.shape)
            self.gui.reset()
        else:
            if self.gui is not None:
                self.gui.close()

        return self.agent_pos

    def _move_agent(self, new_pos: tuple[int, int]):
        """Moves the agent, if possible and updates the 
        corresponding stats.

        Args:
            new_pos: The new position of the agent.
        """

        match self.grid[new_pos]:
            case 0:  # Moved to an empty tile
                self.agent_pos = new_pos
                self.info["agent_moved"] = True
                self.world_stats["total_agent_moves"] += 1
            case 1 | 2:  # Moved to a wall or obstacle
                self.world_stats["total_failed_moves"] += 1
                self.info["agent_moved"] = False
                pass
            case 3:  # Moved to a target tile
                self.agent_pos = new_pos
                self.grid[new_pos] = 0
                if np.sum(self.grid == 3) == 0:
                    self.terminal_state = True
                self.info["target_reached"] = True
                self.world_stats["total_targets_reached"] += 1
                self.info["agent_moved"] = True
                self.world_stats["total_agent_moves"] += 1
                # Otherwise, the agent can't move and nothing happens
            case _:
                raise ValueError(f"Grid is badly formed. It has a value of "
                                 f"{self.grid[new_pos]} at position "
                                 f"{new_pos}.")
        

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """This function makes the agent take a step on the grid.

        Action is provided as integer and values are:
            - 0: Move down
            - 1: Move up
            - 2: Move left
            - 3: Move right
        Args:
            action: Integer representing the action the agent should
                take. 

        Returns:
            0) Current state,
            1) The reward for the agent,
            2) If the terminal state has been reached, and
        """
        
        self.world_stats["total_steps"] += 1
        
        # GUI specific code
        is_single_step = False
        if not self.no_gui:
            start_time = time()
            while self.gui.paused:
                # If the GUI is paused but asking to step, then we step
                if self.gui.step:
                    is_single_step = True
                    self.gui.step = False
                    break
                # Otherwise, we render the current state only
                paused_info = self._reset_info()
                paused_info["agent_moved"] = True
                self.gui.render(self.grid, self.agent_pos, paused_info,
                                0, is_single_step)    

        # Add stochasticity into the agent action
        val = random.random()
        if val > self.sigma:
            actual_action = action
        else:
            actual_action = random.randint(0, 3)
        
        # Make the move
        self.info["actual_action"] = actual_action
        direction = action_to_direction(actual_action)    
        new_pos = (self.agent_pos[0] + direction[0], self.agent_pos[1] + direction[1])

        # Calculate the reward for the agent
        reward = self.reward_fn(self.grid, new_pos)

        self._move_agent(new_pos)
        
        self.world_stats["cumulative_reward"] += reward

        # GUI specific code
        if not self.no_gui:
            time_to_wait = self.target_spf - (time() - start_time)
            if time_to_wait > 0:
                sleep(time_to_wait)
            self.gui.render(self.grid, self.agent_pos, self.info,
                            reward, is_single_step)

        return self.agent_pos, reward, self.terminal_state, self.info

    @staticmethod
    def _default_reward_function(grid, agent_pos) -> float:
        """This is a very simple reward function. Feel free to adjust it.
        Any custom reward function must also follow the same signature, meaning
        it must be written like `reward_name(grid, temp_agent_pos)`.

        Args:
            grid: The grid the agent is moving on, in case that is needed by
                the reward function.
            agent_pos: The position the agent is moving to.

        Returns:
            A single floating point value representing the reward for a given
            action.
        """

        match grid[agent_pos]:
            case 0:  # Moved to an empty tile
                reward = -1
            case 1 | 2:  # Moved to a wall or obstacle
                reward = -5
                pass
            case 3:  # Moved to a target tile
                reward = 10
                # "Illegal move"
            case _:
                raise ValueError(f"Grid cell should not have value: {grid[agent_pos]}.",
                                 f"at position {agent_pos}")
        return reward

    @staticmethod
    def evaluate_agent(grid_fp: Path,
                       agent: BaseAgent,
                       max_steps: int,
                       sigma: float = 0.,
                       agent_start_pos: tuple[int, int] = None,
                       random_seed: int | float | str | bytes | bytearray = 0,
                       show_images: bool = False):
        """Evaluates a single trained agent's performance.

        What this does is it creates a completely new environment from the
        provided grid and does a number of steps _without_ processing rewards
        for the agent. This means that the agent doesn't learn here and simply
        provides actions for any provided observation.

        For each evaluation run, this produces a statistics file in the out
        directory which is a txt. This txt contains the values:
        [ 'total_steps`, `total_failed_moves`]

        Args:
            grid_fp: Path to the grid file to use.
            agent: Trained agent to evaluate.
            max_steps: Max number of steps to take.
            sigma: same as abve.
            agent_start_pos: same as above.
            random_seed: same as above.
            show_images: Whether to show the images at the end of the
                evaluation. If False, only saves the images.
        """

        env = Environment(grid_fp=grid_fp,
                          no_gui=True,
                          sigma=sigma,
                          agent_start_pos=agent_start_pos,
                          target_fps=-1,
                          random_seed=random_seed)
        
        state = env.reset()
        initial_grid = np.copy(env.grid)

        # Add initial agent position to the path
        agent_path = [env.agent_pos]

        for _ in trange(max_steps, desc="Evaluating agent"):
            
            action = agent.take_action(state)
            state, _, terminated, _ = env.step(action)

            agent_path.append(state)

            if terminated:
                break

        env.world_stats["targets_remaining"] = np.sum(env.grid == 3)

        path_image = visualize_path(initial_grid, agent_path)
        file_name = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")

        save_results(file_name, env.world_stats, path_image, show_images)