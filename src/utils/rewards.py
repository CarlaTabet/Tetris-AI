'''Defines the reward function.'''

def calculate_reward(env):
    cleared_lines = len([row for row in env.grid if all(row)])
    height_penalty = max([sum(row) for row in zip(*env.grid)])  # Maximum column height
    if env.state == "gameover":
        return -100  # Heavy penalty for losing
    return cleared_lines * 10 - height_penalty
