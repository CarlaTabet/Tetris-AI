import random
import torch
import numpy as np

'''Implements the DQN agent.'''

class DQNAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize CNN and DQN networks
        self.cnn = config["cnn"]().to(self.device)
        self.policy_net = config["dqn"](config["cnn_output_dim"], len(config["actions"])).to(self.device)
        self.target_net = config["dqn"](config["cnn_output_dim"], len(config["actions"])).to(self.device)

        # Synchronize target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config["lr"])

        # Replay buffer
        self.replay_buffer = config["replay_buffer"]()

    def select_action(self, state, epsilon):
        """
        Selects an action using epsilon-greedy policy.
        Returns the index of the action.
        """
        if random.random() < epsilon:
            # Choose a random action index
            return random.randint(0, len(self.config["actions"]) - 1)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        features = self.cnn(state)
        q_values = self.policy_net(features)
        return torch.argmax(q_values).item()
