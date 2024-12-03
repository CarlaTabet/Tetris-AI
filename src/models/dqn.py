import numpy as np
import torch
import torch.nn as nn
import random

class DQNAgent:
    def __init__(self, cnn, actions, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.1):
        self.cnn = cnn
        self.actions = actions
        self.action_map = {action: idx for idx, action in enumerate(actions)}  # Map actions to indices
        self.inverse_action_map = {idx: action for action, idx in self.action_map.items()}  # Reverse map
        self.optimizer = torch.optim.Adam(self.cnn.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.replay_buffer = []
        self.batch_size = 64

    def select_action(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.actions)  # Exploration
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            q_values = self.cnn(state)
            action = self.actions[torch.argmax(q_values).item()]  # Exploitation

        return self.action_map[action]  # Return the action index

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).unsqueeze(1)
        actions = torch.tensor(actions, dtype=torch.int64)  # Already numerical indices
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.cnn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.cnn(next_states).max(1)[0]
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def train(env, agent, num_episodes=1000):
        for episode in range(num_episodes):
            state = env.get_board_state()
            done = False
            total_reward = 0

            while not done:
                action_index = agent.select_action(state)  # Get numerical action index
                action = agent.inverse_action_map[action_index]  # Convert index back to action
                next_state, reward, done = env.step(action)
                agent.replay_buffer.append((state, action_index, reward, next_state, done))
                state = next_state
                total_reward += reward

                agent.train_step()

            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
            torch.save(agent.cnn.state_dict(), "tetris_cnn.pth")

    # def __init__(self):
    #     super(DeepQNetwork, self).__init__()

    #     self.conv1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True))
    #     self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
    #     self.conv3 = nn.Sequential(nn.Linear(64, 1))

    #     self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x