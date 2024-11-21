from game.tetris import VisualTetris
from rl.agent import DQNAgent
from rl.train import train_agent
from utils.config import CONFIG

'''Entry point for initializing the Tetris environment, agent, and training process.'''

if __name__ == "__main__":

    env = VisualTetris()


    agent = DQNAgent(env, CONFIG)


    try:
        train_agent(env, agent, CONFIG)
    finally:
        env.close()


