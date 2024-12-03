import numpy as np
import pygame
import random
import torch

from models.cnn import TetrisCNN
from models.dqn import DQNAgent

# Define colors
COLORS = [
    (169, 169, 169),  # Gray background for the grid
    (102, 204, 255),  # Cyan
    (255, 178, 102),  # Orange
    (255, 255, 102),  # Yellow
    (178, 102, 255),  # Purple
    (102, 255, 102),  # Green
    (255, 102, 102),  # Red
    (51, 153, 255),   # Blue
]


class Block:
    x = 0
    y = 0

    SHAPES = [
        [[1, 5, 9, 13], [4, 5, 6, 7]],  # Line
        [[4, 5, 9, 10], [2, 6, 5, 9]],  # Z
        [[6, 7, 9, 10], [1, 5, 6, 10]],  # S
        [[1, 2, 5, 9], [0, 4, 5, 6], [1, 5, 9, 8], [4, 5, 6, 10]],  # L
        [[1, 2, 6, 10], [5, 6, 7, 9], [2, 6, 10, 11], [3, 5, 6, 7]],  # Reverse L
        [[1, 4, 5, 6], [1, 4, 5, 9], [4, 5, 6, 9], [1, 5, 6, 9]],  # T
        [[1, 2, 5, 6]],  # Square
    ]

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = random.randint(0, len(self.SHAPES) - 1)
        self.color = self.type + 1  # Color is directly tied to shape index
        self.rotation = 0

    def image(self):
        return self.SHAPES[self.type][self.rotation]

    def rotate(self):
        self.rotation = (self.rotation + 1) % len(self.SHAPES[self.type])


class Tetris:
    def __init__(self, height, width):
        self.level = 2
        self.score = 0
        self.lines = 0
        self.state = "start"
        self.field = [[0 for _ in range(width)] for _ in range(height)]
        self.height = height
        self.width = width
        self.zoom = 25
        self.x_offset = 80
        self.y_offset = 50
        self.active_piece = None

    def spawn_piece(self):
        
        if self.state == "start":  # Only spawn a piece if the game is not over
            self.active_piece = Block(3, 0)

    def check_collision(self):
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.active_piece.image():
                    if (
                        i + self.active_piece.y >= self.height
                        or j + self.active_piece.x >= self.width
                        or j + self.active_piece.x < 0
                        or self.field[i + self.active_piece.y][j + self.active_piece.x] > 0
                    ):
                        return True
        return False

    def freeze_piece(self):
        if self.state == "gameover":  # Do not freeze pieces if the game is over
            return
        block_count = 0  # Count the number of blocks in the active piece
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.active_piece.image():
                    self.field[i + self.active_piece.y][j + self.active_piece.x] = self.active_piece.color
                    block_count += 1
        self.score += block_count  # Increment score by the number of blocks placed
        self.clear_lines()
        self.spawn_piece()
        if self.check_collision():
            self.state = "gameover"

    def move_down(self):
        if self.state == "gameover":  # Prevent movement after game over
            return
        self.active_piece.y += 1
        if self.check_collision():
            self.active_piece.y -= 1
            self.freeze_piece()

    def move_side(self, dx):
        if self.state == "gameover":  # Prevent movement after game over
            return
        self.active_piece.x += dx
        if self.check_collision():
            self.active_piece.x -= dx

    def rotate_piece(self):
        """
        Rotates the current piece clockwise.
        """
        if self.active_piece is None:  # Check if there is an active piece
            return
        prev_rotation = self.active_piece.rotation
        self.active_piece.rotate()
        if self.check_collision():
            self.active_piece.rotation = prev_rotation

    def clear_lines(self):
        full_lines = 0
        for i in range(self.height - 1, -1, -1):  # Iterate from bottom to top
            if 0 not in self.field[i]:  # If the row is full
                del self.field[i]
                self.field.insert(0, [0 for _ in range(self.width)])  # Add a new empty row at the top
                full_lines += 1
        self.lines += full_lines  # Increment the total number of lines cleared

    def draw_grid(self, screen):
        for i in range(self.height):
            for j in range(self.width):
                pygame.draw.rect(
                    screen,
                    COLORS[self.field[i][j]],
                    [
                        self.x_offset + self.zoom * j,
                        self.y_offset + self.zoom * i,
                        self.zoom,
                        self.zoom,
                    ],
                )
                pygame.draw.rect(
                    screen,
                    (200, 200, 200),
                    [
                        self.x_offset + self.zoom * j,
                        self.y_offset + self.zoom * i,
                        self.zoom,
                        self.zoom,
                    ],
                    1,
                )

    def draw_piece(self, screen):
        if self.active_piece and self.state == "start":  # Only draw the piece if the game is not over
            for i in range(4):
                for j in range(4):
                    if i * 4 + j in self.active_piece.image():
                        x = self.x_offset + self.zoom * (j + self.active_piece.x)
                        y = self.y_offset + self.zoom * (i + self.active_piece.y)
                        pygame.draw.rect(
                            screen,
                            COLORS[self.active_piece.color],
                            [x, y, self.zoom, self.zoom],
                        )
                        pygame.draw.rect(
                            screen,
                            (200, 200, 200),
                            [x, y, self.zoom, self.zoom],
                            1,
                        )

    def display_stats(self, screen):
        font = pygame.font.SysFont("Arial", 25, True)
        score_text = font.render(f"Score: {self.score}", True, (0, 0, 0))
        lines_text = font.render(f"Lines: {self.lines}", True, (0, 0, 0))
        screen.blit(score_text, (10, 10))
        screen.blit(lines_text, (10, 40))

    def display_game_over(self, screen):
        font = pygame.font.SysFont("Arial", 50, True)
        over_text = font.render("Game Over", True, (255, 0, 0))
        screen.blit(over_text, (50, 300))

    def get_board_state(self):
        """
        Returns the current grid as a normalized NumPy array with the active piece included.
        """
        state = np.array(self.field, dtype=np.float32)  # Convert to float
        state /= len(COLORS) - 1  # Normalize values between 0 and 1

        if self.active_piece:
            for i in range(4):
                for j in range(4):
                    if i * 4 + j in self.active_piece.image():
                        x = j + self.active_piece.x
                        y = i + self.active_piece.y
                        if 0 <= x < self.width and 0 <= y < self.height:
                            state[y][x] = 1.0  # Active piece cells set to 1
        return state

    def train_rl(self):
        env = Tetris(20, 10)  # Tetris environment
        cnn = TetrisCNN()  # Neural network
        actions = ["LEFT", "RIGHT", "DOWN", "ROTATE"]
        agent = DQNAgent(cnn, actions)

        print("Starting training...")
        DQNAgent.train(env, agent, num_episodes=500)

        # Save the trained model
        torch.save(cnn.state_dict(), "tetris_cnn.pth")
        print("Training complete. Model saved.")

    def count_gaps(self):
        """
        Count the number of gaps (empty cells below filled cells in the grid).
        :return: Total number of gaps in the grid.
        """
        gaps = 0
        for col in range(self.width):
            filled = False
            for row in range(self.height):
                if self.field[row][col] > 0:  # Filled cell
                    filled = True
                elif filled:  # Empty cell below a filled cell
                    gaps += 1
        return gaps

    def step(self, action):
        """
        Executes the given action and updates the game state.
        :param action: Action to take ('LEFT', 'RIGHT', 'DOWN', 'ROTATE').
        :return: Tuple (next_state, reward, done).
        """
        if action == "LEFT":
            self.move_side(-1)
        elif action == "RIGHT":
            self.move_side(1)
        elif action == "DOWN":
            self.move_down()
        elif action == "ROTATE":
            self.rotate_piece()

        if self.active_piece:
            print(f"Active Piece Position: ({self.active_piece.x}, {self.active_piece.y}), Rotation: {self.active_piece.rotation}")
        else:
            print("No active piece.")


        reward = self.calculate_reward()  # Calculate reward based on the current state
        done = self.state == "gameover"  # Check if the game is over
        next_state = self.get_board_state()  # Get the updated board state

        return next_state, reward, done

    def get_max_height(self):
        """
        Get the maximum height of the stack in the grid.
        :return: The maximum height of the stack.
        """
        for row in range(self.height):
            if any(self.field[row]):  # If there's a filled cell in the row
                return self.height - row
        return 0  # Grid is empty

    def calculate_reward(self):
        """
        Calculate reward based on the current game state.
        :return: Reward for the current step.
        """
        if self.state == "gameover":
            return -100  # Heavy penalty for losing

        # Positive reward for points earned (e.g., from clearing lines)
        points_earned = self.score  # Assuming self.score increases per piece placed

        # Penalties for gaps and height
        num_gaps = self.count_gaps()
        max_height = self.get_max_height()
        gap_penalty = num_gaps * 1  # Adjust penalty per gap (e.g., 1 point per gap)
        height_penalty = max_height * 0.5  # Adjust penalty per height (e.g., 0.5 points per height)

        # Final reward
        reward = points_earned - gap_penalty - height_penalty
        return reward


def main():
    pygame.init()
    size = (400, 600)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Modern Tetris")
    clock = pygame.time.Clock()
    fps = 30

    game = Tetris(20, 10)
    game.spawn_piece()

    running = True
    counter = 0

    while running:
        screen.fill((230, 230, 230))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    game.rotate_piece()
                if event.key == pygame.K_DOWN:
                    game.move_down()
                if event.key == pygame.K_LEFT:
                    game.move_side(-1)
                if event.key == pygame.K_RIGHT:
                    game.move_side(1)
                if event.key == pygame.K_SPACE:
                    while not game.check_collision():
                        game.active_piece.y += 1
                    game.active_piece.y -= 1
                    game.freeze_piece()

        if counter % (fps // game.level // 2) == 0:
            if game.state == "start":
                game.move_down()
        counter += 1

        game.draw_grid(screen)
        game.draw_piece(screen)
        game.display_stats(screen)

        if game.state == "gameover":
            game.display_game_over(screen)

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()
    
def main_rl():
    pygame.init()
    size = (400, 600)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("RL Tetris")
    clock = pygame.time.Clock()
    fps = 30

    # Initialize environment and load the trained model
    env = Tetris(20, 10)
    cnn = TetrisCNN()
    cnn.load_state_dict(torch.load("tetris_cnn.pth"))  # Load the saved model
    cnn.eval()  # Set the model to evaluation mode
    actions = ["LEFT", "RIGHT", "DOWN", "ROTATE"]
    agent = DQNAgent(cnn, actions, 0)  # Use greedy policy (no exploration)

    print("Testing the trained model...")

    # Test the trained agent
    running = True
    state = env.get_board_state()
    total_reward = 0

    while running:
        screen.fill((230, 230, 230))

        # RL Agent selects an action
        action = agent.select_action(state)  # Use the trained agent
        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward

        print(f"State: {env.state}, Done: {done}, Reward: {reward}")
        print(f"Selected Action: {action}")

        # Render the game
        env.draw_grid(screen)
        env.draw_piece(screen)
        env.display_stats(screen)

        if env.state == "gameover" or done:
            env.display_game_over(screen)
            print(f"Game Over! Total Reward: {total_reward}")
            running = False

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()

if __name__ == "__main__":
    main_rl()  # Run the testing function
