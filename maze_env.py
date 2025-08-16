import random
from collections import deque

def generate_maze(width, height):
    maze = [[1 for _ in range(width)] for _ in range(height)]

    def carve_passages(cx, cy):
        directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < width and 0 <= ny < height and maze[ny][nx] == 1:
                maze[cy + dy // 2][cx + dx // 2] = 0
                maze[ny][nx] = 0
                carve_passages(nx, ny)

    start_x = random.randrange(1, width, 2)
    start_y = random.randrange(1, height, 2)
    maze[start_y][start_x] = 0
    carve_passages(start_x, start_y)
    return maze

def find_start_goal(maze):
    rows, cols = len(maze), len(maze[0])
    start = goal = None
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == 0:
                start = (r, c)
                break
        if start: break
    for r in reversed(range(rows)):
        for c in reversed(range(cols)):
            if maze[r][c] == 0:
                goal = (r, c)
                break
        if goal: break
    return start, goal

def bfs_shortest_path(maze, start, goal):
    rows, cols = len(maze), len(maze[0])
    queue = deque([start])
    visited = set([start])
    parent = {}
    while queue:
        current = queue.popleft()
        if current == goal:
            path = []
            while current != start:
                path.append(current)
                current = parent[current]
            path.append(start)
            path.reverse()
            return path
        r, c = current
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0:
                neighbor = (nr, nc)
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)
    return None

def place_traps_and_bonuses(maze, start, goal, trap_count=5, bonus_count=5):
    free_cells = [(r,c) for r in range(len(maze)) for c in range(len(maze[0])) if maze[r][c]==0]
    path_cells = bfs_shortest_path(maze, start, goal)
    if not path_cells:
        return [], []
    trap_candidates = [cell for cell in free_cells if cell not in path_cells]
    traps = random.sample(trap_candidates, min(trap_count, len(trap_candidates)))
    remaining = [cell for cell in free_cells if cell not in traps]
    bonuses = random.sample(remaining, min(bonus_count, len(remaining)))
    return traps, bonuses

def step(agent_pos, action, maze, traps, bonuses, collected_bonuses, goal, start,
         ACTION_DICT, STEP_PENALTY, GOAL_REWARD, BONUS_REWARD, TRAP_PENALTY):
    rows, cols = len(maze), len(maze[0])
    dr, dc = ACTION_DICT[action]
    new_r, new_c = agent_pos[0] + dr, agent_pos[1] + dc
    if 0 <= new_r < rows and 0 <= new_c < cols and maze[new_r][new_c]==0:
        agent_pos = [new_r, new_c]

    reward = STEP_PENALTY
    done = False

    if tuple(agent_pos) == goal:
        reward += GOAL_REWARD
        done = True
    elif tuple(agent_pos) in traps:
        reward += TRAP_PENALTY
        agent_pos = list(start)
    elif tuple(agent_pos) in bonuses and tuple(agent_pos) not in collected_bonuses:
        reward += BONUS_REWARD
        collected_bonuses.add(tuple(agent_pos))

    return agent_pos, reward, done, collected_bonuses
