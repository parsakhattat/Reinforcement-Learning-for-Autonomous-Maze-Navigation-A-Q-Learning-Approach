import sys
import os
import pygame
import numpy as np
import random
from maze_env import generate_maze, find_start_goal, place_traps_and_bonuses, step, bfs_shortest_path
from visualization import draw_maze

CELL_SIZE = 40

# RL hyperparameters
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
NUM_EPISODES = 1000

GOAL_REWARD = 50
BONUS_REWARD = 5
TRAP_PENALTY = -10
STEP_PENALTY = -0.1

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ACTION_DICT = {
    0: (-1, 0),  # UP
    1: (1, 0),   # DOWN
    2: (0, -1),  # LEFT
    3: (0, 1),   # RIGHT
}

def choose_action(state, epsilon, Q_table):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(len(ACTIONS)))
    else:
        r, c = state
        return np.argmax(Q_table[r, c])

def get_maze_size():
    while True:
        try:
            size = int(input("Choose maze size (5, 10, 15): "))
            if size in [5, 10, 15]:
                return size + 2  # Add border thickness
            else:
                print("Invalid choice. Please enter 5, 10, or 15.")
        except:
            print("Invalid input. Please enter a number.")

def main():
    global EPSILON, EPSILON_DECAY

    train_mode = input("Do you want visualized training or fast training? (visual/fast): ").strip().lower()
    if train_mode not in ['visual', 'fast']:
        print("Invalid training mode. Defaulting to visualized training.")
        train_mode = 'visual'

    maze_size = get_maze_size()
    QTABLE_FILE = f"qtable_{maze_size-2}.npy"

    maze = generate_maze(maze_size, maze_size)
    start, goal = find_start_goal(maze)
    traps, bonuses = place_traps_and_bonuses(
        maze, start, goal,
        trap_count=max(1, (maze_size-2) // 3),
        bonus_count=max(1, (maze_size-2) // 3)
    )

    playable_size = maze_size - 2
    path = bfs_shortest_path(maze, start, goal)
    MAX_STEPS = max(playable_size * playable_size * 3, len(path) * 7 if path else playable_size * playable_size * 3)
    EPSILON_DECAY = 0.99 if maze_size > 7 else 0.995  # Faster decay for larger mazes
    NUM_EPISODES = 2000 if maze_size > 7 else 1000  # More episodes for larger mazes
    print(f"Max steps per episode: {MAX_STEPS}")
    print(f"Epsilon decay: {EPSILON_DECAY}")
    print(f"DUCTNumber of episodes: {NUM_EPISODES}")

    rows, cols = maze_size, maze_size

    # Q-table logic
    if os.path.exists(QTABLE_FILE):
        Q_table = np.load(QTABLE_FILE)
        print(f"Loaded existing Q-table for maze size {maze_size-2}. Continuing training...")
    else:
        Q_table = np.zeros((rows, cols, len(ACTIONS)))
        print("No saved Q-table found. Starting training from scratch.")
        EPSILON = 1.0  # Reset exploration

    # Initialize Pygame only if needed
    if train_mode == 'visual':
        pygame.init()
        screen = pygame.display.set_mode((cols * CELL_SIZE, rows * CELL_SIZE))
        pygame.display.set_caption("RL Maze")
        clock = pygame.time.Clock()

    successes = 0  # Track goal reaches
    bonus_collections = 0  # Track bonus collections
    for episode in range(1, NUM_EPISODES + 1):
        agent_pos = list(start)
        collected_bonuses = set()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < MAX_STEPS:
            if train_mode == 'visual':
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

            state = tuple(agent_pos)
            action = choose_action(state, EPSILON, Q_table)
            next_pos, reward, done, collected_bonuses = step(
                agent_pos, action, maze, traps, bonuses,
                collected_bonuses, goal, start, ACTION_DICT,
                STEP_PENALTY, GOAL_REWARD, BONUS_REWARD, TRAP_PENALTY
            )
            next_state = tuple(next_pos)

            # Adjust reward for Q-update: 0 if bonus already collected
            q_reward = reward
            if tuple(next_pos) in bonuses and tuple(next_pos) in collected_bonuses:
                q_reward = STEP_PENALTY  # No bonus reward for revisiting
            old_value = Q_table[state[0], state[1], action]
            next_max = np.max(Q_table[next_state[0], next_state[1]])
            new_value = (1 - ALPHA) * old_value + ALPHA * (q_reward + GAMMA * next_max)
            Q_table[state[0], state[1], action] = new_value

            agent_pos = next_pos
            total_reward += reward
            steps += 1

            if reward == BONUS_REWARD:
                bonus_collections += 1
            if done and tuple(agent_pos) == goal:
                successes += 1

            if train_mode == 'visual':
                flash_color = None
                if reward == STEP_PENALTY:
                    flash_color = None
                elif reward > 0:
                    flash_color = (0, 255, 0)
                elif reward < 0:
                    flash_color = (255, 0, 0)

                episode_info = {
                    'episode': episode,
                    'step': steps,
                    'reward': total_reward,
                    'epsilon': EPSILON
                }

                draw_maze(screen, maze, agent_pos, traps, bonuses, collected_bonuses,
                          goal, start, ACTIONS[action], episode_info, CELL_SIZE, flash_color)
                pygame.display.flip()
                clock.tick(30)

        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
        if train_mode == 'visual' or episode % 100 == 0 or episode == 1:
            print(f"Episode {episode} | Reward: {total_reward:.2f} | Steps: {steps} | Epsilon: {EPSILON:.3f} | Successes: {successes} | Bonuses: {bonus_collections}")

    np.save(QTABLE_FILE, Q_table)
    print(f"Training complete. Q-table saved to {QTABLE_FILE}")

if __name__ == "__main__":
    main()