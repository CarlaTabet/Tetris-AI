from models.cnn import TetrisCNN
from models.dqn import DQN
from rl.replay_buffer import ReplayBuffer
from utils.rewards import calculate_reward

'''Centralizes configuration and hyperparameters'''

CONFIG = {
    "cnn": TetrisCNN,
    "cnn_output_dim": 64,
    "dqn": DQN,
    "replay_buffer": lambda: ReplayBuffer(10000),
    "reward_function": calculate_reward,
    "lr": 0.001,
    "epsilon": 1.0,
    "epsilon_decay": 0.995,
    "min_epsilon": 0.1,
    "gamma": 0.99,
    "batch_size": 64,
    "num_episodes": 100,
    "actions": ["LEFT", "RIGHT", "DOWN", "ROTATE"],
    "render_training": False,         # Render the game visually during training
    "render": True,                   # Render during gameplay or debugging
    "target_update": 0.01,
    "save_path": "trained_model.pth"
    # Path to save/load the trained model
}