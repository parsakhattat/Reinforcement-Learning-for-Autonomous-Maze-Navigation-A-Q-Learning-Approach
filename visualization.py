import pygame

def draw_maze(screen, maze, agent_pos, traps, bonuses, collected_bonuses,
              goal, start, action_name, episode_info, cell_size, flash_color=None):
    screen.fill((0,0,0))
    font = pygame.font.SysFont('Arial', cell_size // 3, bold=True)
    text_color_trap = (255,255,255)
    text_color_bonus = (0,0,0)

    rows = len(maze)
    cols = len(maze[0])

    for r in range(rows):
        for c in range(cols):
            rect = pygame.Rect(c*cell_size, r*cell_size, cell_size, cell_size)
            if maze[r][c]==1:
                pygame.draw.rect(screen, (40,40,40), rect)
            else:
                pygame.draw.rect(screen, (200,200,200), rect)

            if (r,c) in traps:
                pygame.draw.rect(screen, (200,0,0), rect)
                text = font.render("-10", True, text_color_trap)
                screen.blit(text, text.get_rect(center=rect.center))
            if (r,c) in bonuses and (r,c) not in collected_bonuses:
                pygame.draw.rect(screen, (0,200,0), rect)
                text = font.render("+5", True, text_color_bonus)
                screen.blit(text, text.get_rect(center=rect.center))
            if (r,c) in collected_bonuses:
                pygame.draw.rect(screen, (150,255,150), rect)

    pygame.draw.rect(screen, (0,0,255), pygame.Rect(start[1]*cell_size,start[0]*cell_size,cell_size,cell_size))
    pygame.draw.rect(screen, (255,215,0), pygame.Rect(goal[1]*cell_size,goal[0]*cell_size,cell_size,cell_size))

    agent_center = (agent_pos[1]*cell_size+cell_size//2, agent_pos[0]*cell_size+cell_size//2)
    pygame.draw.circle(screen, (255,140,0), agent_center, cell_size//3)

    if flash_color:
        flash_surface = pygame.Surface((cell_size,cell_size))
        flash_surface.set_alpha(100)
        flash_surface.fill(flash_color)
        screen.blit(flash_surface, (agent_pos[1]*cell_size, agent_pos[0]*cell_size))

    info_font = pygame.font.SysFont('Arial', 18)
    info_text = f"Episode: {episode_info['episode']}  Step: {episode_info['step']}  Total Reward: {episode_info['reward']:.1f}  Epsilon: {episode_info['epsilon']:.3f}  Action: {action_name}"
    screen.blit(info_font.render(info_text, True, (255,255,255)), (5, rows*cell_size + 5))
