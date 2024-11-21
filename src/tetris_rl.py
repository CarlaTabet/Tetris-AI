import argparse
import torch
from game.tetris import VisualTetris
from models.cnn import TetrisCNN
from models.dqn import DQN
from rl.agent import DQNAgent
from rl.train import train_agent
from utils.config import CONFIG


def train_model():
    """
    Train the RL agent and save the model.
    """
    env = VisualTetris()
    agent = DQNAgent(env, CONFIG)

    try:
        print("Starting training...")
        train_agent(env, agent, CONFIG)
        print(f"Training complete. Model saved at {CONFIG['save_path']}.")
    finally:
        env.close()


def play_model(render=True):
    """
    Load a trained model and watch it play Tetris.
    """
    env = VisualTetris()

    try:
        # Load the trained model
        cnn = CONFIG["cnn"]()
        policy_net = CONFIG["dqn"](CONFIG["cnn_output_dim"], len(CONFIG["actions"]))
        policy_net.load_state_dict(torch.load(CONFIG["save_path"]))
        policy_net.eval()

        state = env.get_board_state()
        done = False

        while not done:
            if render:
                env.render()

            # Get the best action from the policy network
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            features = cnn(state_tensor)
            q_values = policy_net(features)
            action_index = torch.argmax(q_values).item()

            # Map action index to action string
            action = CONFIG["actions"][action_index]

            # Step in the environment
            next_state, _, done = env.step(action)
            state = next_state

            print(f"State before action: {state}")
            print(f"Selected action: {CONFIG['actions'][action_index]}")
            print(f"State after action: {next_state}, Done: {done}")

    finally:
        env.close()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train or Play Tetris with Reinforcement Learning")
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["train", "play"],
        required=True,
        help="Mode: 'train' to train the model, 'play' to watch gameplay",
    )
    parser.add_argument(
        "-r",
        "--render",
        action="store_true",
        help="Render the game visually during training or playing",
    )

    args = parser.parse_args()

    if args.mode == "train":
        CONFIG["render_training"] = args.render
        train_model()
    elif args.mode == "play":
        play_model(render=args.render)
