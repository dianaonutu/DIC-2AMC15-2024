# DIC-2AMC15-2024

Welcome to Data Intelligence Challenge-2AMC15!
This is the repository containing the challenge environment code.

## Quickstart

1. Create a virtual environment for this course with Python >= 3.10. Using conda, you can do: `conda create -n dic2024 python=3.11`. Use `conda activate dic2024` to activate it `conda deactivate` to deactivate it.
2. Clone this repository into the local directory you prefer `git clone https://github.com/dianaonutu/DIC-2AMC15-2024.git`.
3. Install the required packages `pip install -r requirements.txt`. Now, you are ready to use the simulation environment! :partying_face:	
4. Run `$ python train.py grid_configs/testroom.grd` to start training!

`train.py` is just an example training script. Inside this file, initialize the agent you want to train and evaluate. Feel free to modify it as necessary. Its usage is:

```bash
usage: train.py [-h] [--no_gui] [--sigma SIGMA] [--fps FPS] [--iter ITER]
                [--random_seed RANDOM_SEED] 
                GRID [GRID ...]

DIC Reinforcement Learning Trainer.

positional arguments:
  GRID                  Paths to the grid file to use. There can be more than
                        one.
options:
  -h, --help                 show this help message and exit
  --no_gui                   Disables rendering to train faster
  --sigma SIGMA              Sigma value for the stochasticity of the environment.
  --fps FPS                  Frames per second to render at. Only used if no_gui is not set.
  --iter ITER                Number of iterations to go through.
  --random_seed RANDOM_SEED  Random seed value for the environment.
```

## Code guide

The code is made up of 2 modules: 

1. `agent`
2. `world`

### The `agent` module

The `agent` module contains the `BaseAgent` class as well as some benchmark agents you may want to test against.

The `BaseAgent` is an abstract class and all RL agents for DIC must inherit from/implement it.
If you know/understand class inheritence, skip the following section:

#### `BaseAgent` as an abstract class
Think of this like how all models in PyTorch start like 

```python
class NewModel(nn.Module):
    def __init__(self):
        super().__init__()
    ...
```

In this case, `NewModel` inherits from `nn.Module`, which gives it the ability to do back propagation, store parameters, etc. without you having to manually code that every time.
It also ensures that every class that inherits from `nn.Module` contains _at least_ the `forward()` method, which allows a forward pass to actually happen.

In the case of your RL agent, inheriting from `BaseAgent` guarantees that your agent implements `update()` and `take_action()`.
This ensures that no matter what RL agent you make and however you code it, the environment and training code can always interact with it in the same way.
Check out the benchmark agents to see examples.

### The `world` module

The world module contains:
1. `grid_creator.py`
2. `environment.py`
3. `grid.py`
4. `gui.py`

#### Grid creator
Run this file to create new grids.

```bash
$ python grid_creator.py
```

This will start up a web server where you create new grids, of different sizes with various elements arrangements.
To view the grid creator itself, go to `127.0.0.1:5000`.
All levels will be saved to the `grid_configs/` directory.


#### The Environment

The `Environment` is very important because it contains everything we hold dear, including ourselves [^1].
It is also the name of the class which our RL agent will act within. Most of the action happens in there.

The main interaction with `Environment` is through the methods:

- `Environment()` to initialize the environment
- `reset()` to reset the environment
- `step()` to actually take a time step with the environment
- `Environment().evaluate_agent()` to evaluate the agent after training.

[^1]: In case you missed it, this sentence is a joke. Please do not write all your code in the `Environment` class.

#### The Grid

The `Grid` class is the the actual representation of the world on which the agent moves. It is a 2D Numpy array.

#### The GUI

The Graphical User Interface provides a way for you to actually see what the RL agent is doing.
While performant and written using PyGame, it is still about 1300x slower than not running a GUI.
Because of this, we recommend using it only while testing/debugging and not while training.
