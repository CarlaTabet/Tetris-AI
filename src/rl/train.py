import torch
import numpy as np

'''Handles the training loop and DQN updates.'''
def train_agent(env, agent, config):
    epsilon = config["epsilon"]
    max_steps = config.get("max_steps_per_episode", 10)

    for episode in range(config["num_episodes"]):
        state = env.get_board_state()
        done = False
        step_count = 0
        cumulative_reward = 0

        while not done and step_count < max_steps:
            step_count += 1

            # Render environment if enabled
            if config["render"]:
                env.render()

            # Select and execute action
            action = agent.select_action(state, epsilon)
            next_state, reward, done = env.step(action)

            # Track cumulative reward
            cumulative_reward += reward

            # Store transition in replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            # Perform training step
            if len(agent.replay_buffer) > config["batch_size"]:
                train_step(agent, config)

        # Decay epsilon
        epsilon = max(config["min_epsilon"], epsilon * config["epsilon_decay"])

        # Log episode information
        print(f"Episode {episode + 1}/{config['num_episodes']} | Reward: {cumulative_reward} | Steps: {step_count} | Epsilon: {epsilon:.4f}")

        if step_count >= max_steps:
            print("Reached maximum steps for episode.")

    # Save the trained model
    torch.save(agent.policy_net.state_dict(), config["save_path"])
    print(f"Model saved to {config['save_path']}")


def train_step(agent, config):
    """
    Performs a single training step on a batch from the replay buffer.
    """
    batch = agent.replay_buffer.sample(config["batch_size"])
    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert to PyTorch tensors
    states = torch.tensor(np.array(states), dtype=torch.float32).unsqueeze(1).to(agent.device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).unsqueeze(1).to(agent.device)
    actions = torch.tensor(actions, dtype=torch.int64).to(agent.device)  # Action indices
    rewards = torch.tensor(rewards, dtype=torch.float32).to(agent.device)
    dones = torch.tensor(dones, dtype=torch.float32).to(agent.device)

    # Compute Q-values and targets
    features = agent.cnn(states)
    q_values = agent.policy_net(features).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_features = agent.cnn(next_states)
        max_next_q_values = agent.target_net(next_features).max(1)[0]
        targets = rewards + config["gamma"] * max_next_q_values * (1 - dones)

    # Compute loss and update
    loss = torch.nn.functional.mse_loss(q_values, targets)
    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()

    # Periodically update target network
    if np.random.rand() < config["target_update"]:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())

