# Reinforcement Learning for Autonomous Maze Navigation: A Q-Learning Approach

## Project Overview
Inspired by the challenge of enabling machines to navigate complex environments through trial and error, this project presents a Q-learning-based reinforcement learning (RL) agent designed to traverse randomly generated mazes. The agent learns to avoid traps, collect bonuses, and reach a goal, demonstrating the application of RL in a dynamic setting. Mazes are generated in sizes of 5x5, 10x10, or 15x15 (playable areas, with borders expanding to 7x7, 12x12, or 17x17) using a recursive backtracking algorithm to ensure solvability.

A Q-learning algorithm with epsilon-greedy exploration is implemented, with hyperparameters dynamically adjusted based on maze size. Two training modes are supported: visualized mode, rendering navigation in real-time via Pygame, and fast mode, optimizing computation. Pre-trained Q-tables (`qtable_5.npy`, `qtable_10.npy`, `qtable_15.npy`) are provided for immediate use, with the option to train new Q-tables.

## Features
- Random mazes generated with walls, traps, bonuses, start, and goal positions using recursive backtracking.
- Q-learning algorithm implemented with epsilon-greedy exploration for optimal navigation policies.
- Two training modes provided:
  - *Visual Mode*: Real-time rendering of agent movement and episode statistics via Pygame.
  - *Fast Mode*: Computation-focused training without visualization.
- Reward structure defined:
  - Goal reached: +50 (ends episode).
  - Bonus collected: +5 (once per cell, with a fix to prevent revisiting).
  - Trap hit: -10 (resets to start).
  - Step: -0.1.
- Visualization designed with Pygame, displaying walls (dark), paths (light gray), traps (red, -10), bonuses (green, +5), collected bonuses (light green), start (blue), goal (gold), and agent (orange circle), with green/red flashes for rewards.
- Q-tables saved/loaded (`qtable_5.npy`, `qtable_10.npy`, `qtable_15.npy`) for continued training or immediate use.
- Hyperparameters adjusted dynamically based on maze size.

## Project Structure
- `main.py`: Core implementation of Q-learning, maze setup, and training logic (visual/fast modes).
- `visualization.py`: Rendering of maze and agent with Pygame, including episode statistics and reward feedback.
- `maze_env.py`: Generation of mazes, placement of start/goal/traps/bonuses, and step mechanics.
- `qtable_5.npy`, `qtable_10.npy`, `qtable_13.npy`: Pre-trained Q-tables for 5x5, 10x10, and 15x15 mazes.

## Installation
To set up the project, the following steps are recommended:
1. Clone the repository:
   ```bash
   git clone https://github.com/parsakhattat/Reinforcement-Learning-for-Autonomous-Maze-Navigation-A-Q-Learning-Approach
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure Python 3.6 or higher is installed.

## Usage
The main script is executed as follows:
```bash
python main.py
```
1. Select training mode:
   - `visual`: Displays real-time agent navigation with episode statistics (episode, step, reward, epsilon).
   - `fast`: Trains efficiently without visualization, printing stats every 100 episodes or for episode 1.
2. Choose maze size (`5`, `10`, or `15` for 5x5, 10x10, or 15x15 playable areas).

A random maze is generated, and the agent is trained for 1000 episodes (5x5) or 2000 episodes (10x10/15x15), with the Q-table saved as `qtable_{size}.npy`.

- **Pre-trained Q-Tables**: Files `qtable_5.npy`, `qtable_10.npy`, and `qtable_15.npy` are included. Retain these files in the project directory to use pre-trained policies in visual mode for immediate navigation. Remove the relevant Q-table to train from scratch.
- **Training New Q-Tables**: Delete the corresponding Q-table file (e.g., `qtable_15.npy`) to start fresh training on a new maze.
- **Exiting Visual Mode**: Close the Pygame window to terminate training.

## Performance
The RL agent achieves robust performance across maze sizes. An example from fast-mode training on a 15x15 maze (2000 episodes, continuing from `qtable_15.npy`) demonstrates:
- **Success Rate**: 1983/2000 goal reaches (99.15%) by episode 2000.
- **Average Reward**: ~46.3-46.4 (goal reward of 50 minus ~36-37 steps at -0.1 each).
- **Steps per Episode**: ~36-37, significantly below the maximum of 675.
- **Bonuses**: 0 collected, indicating goal prioritization, likely due to maze layout or the bonus-chasing fix.
- **Training Progress**: Successes increased steadily (e.g., 83 at episode 100, 983 at episode 1000, 1983 at episode 2000).

Comparable performance is observed for 5x5 and 10x10 mazes, with success rates typically exceeding 96% and rewards approaching 50, depending on maze layout. Pre-trained Q-tables enable immediate high-performance navigation in visual mode, ideal for demonstrations.

## Technical Details
### Maze Generation (`maze_env.py`)
- A recursive backtracking algorithm generates mazes with walls (1) and paths (0).
- Start is set as the first open cell (top-left), goal as the last (bottom-right).
- Traps and bonuses are placed randomly on paths, avoiding the shortest path (via BFS). Count: `max(1, (maze_size-2) // 3)` (e.g., ~3 for 10x10).
- BFS calculates the shortest path to determine max steps.

### Q-Learning Algorithm (`main.py`)
- **State**: Agent position (row, col).
- **Actions**: UP, DOWN, LEFT, RIGHT.
- **Q-Table**: 3D NumPy array (`rows x cols x 4 actions`).
- **Update Rule**: `Q(s, a) = (1 - α) * Q(s, a) + α * (r + γ * max(Q(s', a')))` with `α = 0.1`, `γ = 0.9`.
- **Exploration**: Epsilon-greedy, starting at `ε = 1.0`, decaying to `ε = 0.01` with `EPSILON_DECAY = 0.995` (5x5) or `0.99` (10x10/15x15).
- **Episodes**: 1000 (5x5), 2000 (10x10/15x15).
- **Max Steps**: `max((maze_size-2)^2 * 3, shortest_path_length * 7)`.
- **Reward Fix**: Reward set to -0.1 for revisiting collected bonuses during Q-table updates.

### Visualization (`visualization.py`)
- Pygame renders walls (dark), paths (light gray), traps (red, -10), bonuses (green, +5), collected bonuses (light green), start (blue), goal (gold), and agent (orange circle).
- Green/red flashes indicate positive/negative rewards.
- Episode, step, total reward, epsilon, and action are displayed below the maze.

## Further Improvements
Several enhancements could be explored to extend the project’s capabilities:

- **Hyperparameter Tuning**: Learning rate (`α`), discount factor (`γ`), and epsilon decay could be optimized using grid search to improve convergence speed and performance.
- **Advanced Algorithms**: Deep Q-learning or SARSA could be implemented to handle larger mazes or continuous state spaces, enhancing scalability.
- **Dynamic Rewards**: Trap and bonus rewards could be adjusted dynamically based on maze size or distance to the goal to balance exploration and goal-seeking.
- **Visualization Enhancements**: Additional metrics (e.g., success rate, average reward over time) could be plotted in real-time using libraries like Matplotlib, improving interpretability.
- **Maze Complexity**: Features like moving traps or multiple goals could be introduced to increase challenge and realism.
These improvements would further demonstrate the project’s potential and adaptability to advanced RL scenarios.

## Training Recommendations
To ensure robust performance across maze layouts, multiple training runs are recommended:
- **5x5**: 1-2 runs (1000 episodes each), targeting >800 goal reaches per run.
- **10x10**: 2-3 runs (2000 episodes each), targeting >1600 goal reaches per run.
- **15x15**: 3-5 runs (2000 episodes each), targeting >1600 goal reaches per run.

The corresponding Q-table file should be deleted before each run to train on a new maze. Alternatively, pre-trained Q-tables can be used for immediate results in visual mode.

## Motivation
Inspired by the challenge of enabling machines to learn through trial and error, this project was developed to explore reinforcement learning in a tangible, interactive context. The process of designing the Q-learning algorithm, optimizing maze generation, and creating an engaging visualization deepened the understanding of AI and programming. This work aims to demonstrate the potential of RL in solving complex problems and inspire further exploration in the field.

## License
Licensed under the MIT License. Use, modification, and distribution are permitted with proper attribution.

## Acknowledgments
This project was created to demonstrate proficiency in reinforcement learning, Python, NumPy, and Pygame. Feedback is welcomed to further refine the implementation.